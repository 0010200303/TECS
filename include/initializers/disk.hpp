#ifndef TECS_INITIALIZERS_DISK_INIT_HPP
#define TECS_INITIALIZERS_DISK_INIT_HPP

#include "../utils/utils.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents at uniformly random positions inside a 2D disk.
   *
   * Agents are distributed uniformly by area inside a circle. The disk radius
   * is estimated from the agent count and the desired neighbour spacing.
   */
  template<typename SimulationConfig>
  struct RandomDisk {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    static_assert(dimensions >= 2, "RandomDisk at least requires 2-dimensional vectors");

    VectorView positions_view;
    /// Desired distance between neighbouring agents.
    Scalar distance_nb;

    /**
     * @param positions              View to fill with positions.
     * @param distance_to_neighbour  Approximate spacing between agents.
     */
    template<typename ViewType>
    RandomDisk(ViewType positions, Scalar distance_to_neighbour)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour) { }

    /// @brief Sample a random position inside the disk for agent @p i.
    INIT_OP {
      const Scalar r_max = Kokkos::sqrt(static_cast<Scalar>(positions_view.extent(0))) * distance_nb / Scalar(2.0);
      const Scalar r  = r_max * Kokkos::sqrt(static_cast<Scalar>(generator.drand()));
      const Scalar phi = static_cast<Scalar>(generator.drand()) * Scalar(2.0) * Kokkos::numbers::pi_v<Scalar>;

      positions_view(i)[0] = r * Kokkos::sin(phi);
      positions_view(i)[1] = r * Kokkos::cos(phi);
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_DISK_INIT_HPP
