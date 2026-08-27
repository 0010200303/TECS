#ifndef TECS_INITIALIZERS_RECTANGLE_INIT_HPP
#define TECS_INITIALIZERS_RECTANGLE_INIT_HPP

#include "../utils/utils.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents on a regular rectangular grid (2D) with
   *        a staggered (triangular) layout.
   *
   * Odd rows are shifted by half the neighbour distance, creating a
   * hexagonal-like arrangement with row spacing @f$ d \cdot \sqrt{3}/2 @f$.
   */
  template<typename SimulationConfig>
  struct RegularRectangle {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    static_assert(dimensions >= 2, "RegularRectangle at least requires 2-dimensional vectors");

    VectorView positions_view;
    /// Distance between neighbouring agents.
    Scalar distance_nb;
    /// Number of agents per row (x-direction).
    unsigned int nx;

    /**
     * @param positions              View to fill with positions.
     * @param distance_to_neighbour  Spacing between adjacent agents.
     * @param nx_                    Number of columns per row.
     */
    template<typename ViewType>
    RegularRectangle(ViewType positions, Scalar distance_to_neighbour, unsigned int nx_)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour)
      , nx(nx_) { }

    /// @brief Compute the grid position for agent @p i.
    INIT_OP {
      const unsigned int row = i / nx;
      const unsigned int col = i % nx;

      // row spacing = sqrt(d_nb^2 - (d_nb/2)^2) = d_nb * sqrt(3)/2
      const Scalar row_height = distance_nb * Scalar(0.5) * Kokkos::sqrt(Scalar(3.0));
      const Scalar row_offset = (row % 2 != 0) ? (distance_nb * Scalar(0.5)) : Scalar(0.0);

      positions_view(i)[0] = row_offset + Scalar(col) * distance_nb;
      positions_view(i)[1] = Scalar(row) * row_height;
    }
  };
} // namespace tecs::initializers

#endif // TECS_INITIALIZERS_RECTANGLE_INIT_HPP
