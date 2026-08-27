#ifndef TECS_INITIALIZERS_HEXAGON_INIT_HPP
#define TECS_INITIALIZERS_HEXAGON_INIT_HPP

#include <Kokkos_MathematicalConstants.hpp>

#include "../utils/utils.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents on a regular hexagonal lattice (2D).
   *
   * Agents are arranged in concentric hexagonal rings around a central agent.
   */
  template<typename SimulationConfig>
  struct RegularHexagon {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    static_assert(dimensions >= 2, "RegularHexagon at least requires 2-dimensional vectors");

    VectorView positions_view;
    /// Distance between neighbouring agents.
    Scalar distance_nb;

    /**
     * @param positions              View to fill with positions.
     * @param distance_to_neighbour  Spacing between adjacent agents.
     */
    template<typename ViewType>
    RegularHexagon(ViewType positions, Scalar distance_to_neighbour)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour) { }

    /// @brief Compute the hexagonal lattice position for agent @p i.
    INIT_OP {
      if (i == 0)
        return;

      auto& position = positions_view(i);

      // figure out current ring
      unsigned int remaining = i - 1;
      unsigned int ring = 1;
      while (remaining >= 6 * ring) {
        remaining -= 6 * ring;
        ++ring;
      }

      const Scalar beta = Scalar(Kokkos::numbers::pi_v<Scalar>) / Scalar(3);
      const unsigned int side = remaining / ring;
      const unsigned int step = remaining % ring;

      const Scalar radius = distance_nb * Scalar(ring);
      const Scalar t = Scalar(step) / Scalar(ring);

      Vector start{Scalar(0)};
      start[0] = -radius * Kokkos::sin(beta * Scalar(side));
      start[1] =  radius * Kokkos::cos(beta * Scalar(side));

      Vector end{Scalar(0)};
      end[0] = -radius * Kokkos::sin(beta * Scalar(side + 1));
      end[1] =  radius * Kokkos::cos(beta * Scalar(side + 1));

      position = Vector{Scalar(0)};
      position[0] = start[0] + (end[0] - start[0]) * t;
      position[1] = start[1] + (end[1] - start[1]) * t;
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_HEXAGON_INIT_HPP
