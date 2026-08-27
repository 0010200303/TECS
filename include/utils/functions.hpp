#ifndef TECS_UTILS_FUNCTION_HPP
#define TECS_UTILS_FUNCTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "utils.hpp"
#include "../types/vector.hpp"
#include "../types/view.hpp"

namespace tecs::utils {

  /**
   * @brief Stores per-dimension minimum and maximum values of a set of points.
   *
   * Used as a reduction target in `get_bounds()`.
   *
   * @tparam Vector A vector type (e.g. `VectorN<Scalar, N>`).
   */
  template<typename Vector>
  struct Bounds {
    Vector min;  ///< Minimum coordinate in each dimension.
    Vector max;  ///< Maximum coordinate in each dimension.
  };

  /**
   * @brief Kokkos reduction functor that finds the minimum value in each
   *        dimension across a set of points.
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points (default: `View<Vector>`).
   */
  template<typename Vector, typename PointsView = View<Vector>>
  struct MinPerDimension {
    EXTRACT_VECTOR(Vector)

    PointsView points_view;

    MinPerDimension(const PointsView& points_view_) : points_view(points_view_) { }

    /**
     * @brief Update the per-dimension minimum with point @p i.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Vector& local) const {
      Vector point = read_element(points_view, i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value < local[d])
          local[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Vector& value) const {
      value = Vector(Kokkos::Experimental::finite_max_v<Scalar>);
    }

    KOKKOS_INLINE_FUNCTION
    void join(Vector& dst, const Vector& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src[d] < dst[d])
          dst[d] = src[d];
      }
    }
  };

  /**
   * @brief Kokkos reduction functor that finds the maximum value in each
   *        dimension across a set of points.
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points (default: `View<Vector>`).
   */
  template<typename Vector, typename PointsView = View<Vector>>
  struct MaxPerDimension {
    EXTRACT_VECTOR(Vector)

    PointsView points_view;

    MaxPerDimension(const PointsView& points_view_) : points_view(points_view_) { }

    /**
     * @brief Update the per-dimension maximum with point @p i.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Vector& local) const {
      Vector point = read_element(points_view, i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value > local[d])
          local[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Vector& value) const {
      value = Vector(Kokkos::Experimental::finite_min_v<Scalar>);
    }

    KOKKOS_INLINE_FUNCTION
    void join(Vector& dst, const Vector& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src[d] > dst[d])
          dst[d] = src[d];
      }
    }
  };

  /**
   * @brief Kokkos reduction functor that computes both min and max in each
   *        dimension at once (more efficient than separate passes).
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points (default: `View<Vector>`).
   */
  template<typename Vector, typename PointsView = View<Vector>>
  struct BoundsPerDimension {
    EXTRACT_VECTOR(Vector)

    PointsView points_view;

    BoundsPerDimension(const PointsView& points_view_) : points_view(points_view_) { }

    /**
     * @brief Update both min and max with point @p i.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Bounds<Vector>& local) const {
      Vector point = read_element(points_view, i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value < local.min[d])
          local.min[d] = value;
        if (value > local.max[d])
          local.max[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Bounds<Vector>& value) const {
      value = Bounds<Vector>(
        Vector(Kokkos::Experimental::finite_max_v<Scalar>),
        Vector(Kokkos::Experimental::finite_min_v<Scalar>)
      );
    }

    KOKKOS_INLINE_FUNCTION
    void join(Bounds<Vector>& dst, const Bounds<Vector>& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src.min[d] < dst.min[d])
          dst.min[d] = src.min[d];
        if (src.max[d] > dst.max[d])
          dst.max[d] = src.max[d];
      }
    }
  };

  /**
   * @brief Compute the minimum coordinate across all points, per dimension.
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points.
   * @param points_view  The set of points to scan.
   * @return A vector where each component is the minimum in that dimension.
   */
  template<typename Vector, typename PointsView = View<Vector>>
  Vector get_min_bounds(const PointsView& points_view) {
    Vector result;
    Kokkos::parallel_reduce(
      "utils_get_min_bounds",
      points_view.extent(0),
      MinPerDimension<Vector, PointsView>(points_view),
      result
    );
    return result;
  }

  /**
   * @brief Compute the maximum coordinate across all points, per dimension.
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points.
   * @param points_view  The set of points to scan.
   * @return A vector where each component is the maximum in that dimension.
   */
  template<typename Vector, typename PointsView = View<Vector>>
  Vector get_max_bounds(const PointsView& points_view) {
    Vector result;
    Kokkos::parallel_reduce(
      "utils_get_max_bounds",
      points_view.extent(0),
      MaxPerDimension<Vector, PointsView>(points_view),
      result
    );
    return result;
  }

  /**
   * @brief Compute both min and max bounds in a single parallel reduction.
   *
   * More efficient than calling `get_min_bounds()` and `get_max_bounds()`
   * separately.
   *
   * @tparam Vector      Vector type for one point.
   * @tparam PointsView  View type containing the points.
   * @param points_view  The set of points to scan.
   * @return A `Bounds` struct with the min and max per dimension.
   */
  template<typename Vector, typename PointsView = View<Vector>>
  Bounds<Vector> get_bounds(const PointsView& points_view) {
    Bounds<Vector> result;
    Kokkos::parallel_reduce(
      "utils_get_bounds",
      points_view.extent(0),
      BoundsPerDimension<Vector, PointsView>(points_view),
      result
    );
    return result;
  }
} // namespace tecs::utils

#endif // TECS_UTILS_FUNCTION_HPP
