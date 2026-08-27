#ifndef TECS_INITIALIZERS_LINE_INIT_HPP
#define TECS_INITIALIZERS_LINE_INIT_HPP

#include "../utils/utils.hpp"

namespace tecs::initializers {

  /**
   * @brief Place agents in a straight line with equal spacing across all dimensions.
   *
   * Agent @p i is placed at @f$ (i \cdot \text{distance}, 0, 0, \dots) @f$.
   */
  template<typename SimulationConfig>
  struct Line {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    /// Spacing between consecutive agents.
    const Scalar distance;

    /**
     * @param positions   View to fill with positions.
     * @param distance_   Distance between agents.
     */
    Line(VectorView positions, const Scalar distance_) 
      : positions_view(positions)
      , distance(distance_) { }

    /// @brief Set position of agent @p i along the line.
    INIT_OP {
      positions_view(i) = Vector(Scalar(i) * distance);
    }
  };
} // namespace tecs::initalizers

#endif // TECS_INITIALIZERS_LINE_INIT_HPP
