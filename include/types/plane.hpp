#ifndef TECS_TYPES_PLANE_HPP
#define TECS_TYPES_PLANE_HPP

#include <Kokkos_Core.hpp>

#include "vector.hpp"

namespace tecs {

  /**
   * @brief A simple Plane.
   *
   * Stores a plane (or line/hyperplane in N dimensions) using `dimensions + 1`
   * coefficients: the first `dimensions` entries define the unit normal vector,
   * and the last entry is the signed distance from the origin (the **offset** or
   * **d**). The plane equation is:
   *
   * @f[
   *   n_0 \cdot x_0 + n_1 \cdot x_1 + \dots + n_{d-1} \cdot x_{d-1} + d = 0
   * @f]
   *
   * where @f$ \mathbf{n} @f$ is the unit normal and @f$ d @f$ is the offset.
   *
   * @tparam Scalar     Floating-point type (e.g. `double`, `float`).
   * @tparam dimensions The number of spatial dimensions (1 = point/line,
   *                    2 = line, 3 = plane, ...).
   * @tparam Align      Memory alignment (defaults to the scalar alignment).
   *
   * Convenience aliases `Plane1`, `Plane2`, `Plane3`, `Plane4` are provided
   * for the most common dimensionalities.
   */
  template<typename Scalar, unsigned int dimensions, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) PlaneN {
    using Vector = VectorN<Scalar, dimensions>;

    /**
     * The plane coefficients: `[nx, ny, ..., d]`.
     * The normal is always stored as a unit vector.
     */
    VectorN<Scalar, dimensions + 1> coefficients;

    /**
     * @brief Default constructor: creates the plane @f$ z = 0 @f$
     *        (or the equivalent in the last dimension).
     */
    KOKKOS_INLINE_FUNCTION
    PlaneN() {
      coefficients[dimensions - 1] = Scalar(1);
    }

    /**
     * @brief Construct a plane from its coefficients and normalise it.
     *
     * The first `dimensions` values define the (unnormalised) normal and the
     * last value is the offset. The coefficients are automatically normalised
     * so that the normal becomes a unit vector.
     *
     * @param scalars The coefficients @f$ (n_0, n_1, \dots, n_{d-1}, d) @f$.
     */
    template<typename... Scalars>
    KOKKOS_INLINE_FUNCTION
    PlaneN(const Scalars... scalars) : coefficients(static_cast<Scalar>(scalars)...) {
      Scalar squared_sum = Scalar(0);
      for (unsigned int i = 0; i < dimensions; ++i)
        squared_sum += coefficients[i] * coefficients[i];

      Scalar inv_norm = Scalar(1) / Kokkos::sqrt(squared_sum);
      for (unsigned int i = 0; i < dimensions + 1; ++i)
        coefficients[i] *= inv_norm;
    }

    /**
     * @brief Compute the signed distance from a point to the plane.
     *
     * A positive value means the point is on the side of the normal,
     * a negative value means it is on the opposite side.
     *
     * @param point The query point.
     * @return Signed distance.
     */
    KOKKOS_INLINE_FUNCTION
    Scalar signed_distance(const Vector& point) const {
      Scalar result = coefficients[dimensions];
      for (unsigned int i = 0; i < dimensions; ++i)
        result += coefficients[i] * point[i];
      return result;
    }

    /**
     * @brief Return the (already normalised) unit normal of the plane.
     */
    KOKKOS_INLINE_FUNCTION
    Vector normal() const {
      Vector normal;
      for (unsigned int i = 0; i < dimensions; ++i)
        normal[i] = coefficients[i];
      return normal;
    }
  };

  /// @brief 1D plane (a point).
  template<typename Scalar>
  using Plane1 = PlaneN<Scalar, 1>;

  /// @brief 2D plane (a line).
  template<typename Scalar>
  using Plane2 = PlaneN<Scalar, 2>;

  /// @brief 3D plane.
  template<typename Scalar>
  using Plane3 = PlaneN<Scalar, 3>;

  /// @brief 4D plane (a hyperplane).
  template<typename Scalar>
  using Plane4 = PlaneN<Scalar, 4>;
} // namespace tecs

#endif // TECS_TYPES_PLANE_HPP
