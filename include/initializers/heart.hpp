// https://www.researchgate.net/publication/330496498_New_approach_to_Generate_Re-Generate_Encode_and_Compress_3D_structures_using_polynomial_equations

#ifndef TECS_INITIALIZERS_HEART_INIT_HPP
#define TECS_INITIALIZERS_HEART_INIT_HPP

#include "../utils/utils.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents inside a 3D heart-shaped volume.
   *
   * Agents are sampled uniformly inside a heart-shaped surface described by
   * a polynomial implicit equation, using rejection sampling over a bounding
   * cube. The volume is then scaled by the neighbour spacing factor @p scale.
   */
  template<typename SimulationConfig>
  struct RandomHeart {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    static_assert(dimensions >= 3, "RandomHeart requires at least 3-dimensional vectors");

    VectorView positions_view;
    /// Desired scale of the heart.
    Scalar scale;

    /**
     * @param positions  View to fill with positions.
     * @param scale_     Desired scale of the heart.
     */
    template<typename ViewType>
    RandomHeart(ViewType positions, Scalar scale_)
      : positions_view(positions)
      , scale(scale_) { }

    KOKKOS_INLINE_FUNCTION
    Scalar _check_heart(const Scalar x, const Scalar y, const Scalar z) const {
      return Kokkos::pow(x * x + 9.0 / 4.0 * y * y + z * z - 1.0, 3)
        - x * x * z * z * z
        - 9.0 / 80.0 * y * y * z * z * z;
    }

    /// @brief Sample a random position inside the heart volume for agent @p i.
    INIT_OP {
      Scalar x, y, z;
      do {
        x = static_cast<Scalar>(generator.drand(-2.0, 2.0));
        y = static_cast<Scalar>(generator.drand(-2.0, 2.0));
        z = static_cast<Scalar>(generator.drand(-2.0, 2.0));
      } while (_check_heart(x, y, z) > Scalar(0.0));

      positions_view(i)[0] = x * scale;
      positions_view(i)[1] = y * scale;
      positions_view(i)[2] = z * scale;
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_HEART_INIT_HPP
