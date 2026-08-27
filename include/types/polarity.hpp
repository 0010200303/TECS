#ifndef TECS_TYPES_POLARITY_HPP
#define TECS_TYPES_POLARITY_HPP

#include "vector.hpp"

namespace tecs {

  /**
   * @brief Represents a Polarity in 3D using spherical coordinates $(\theta, \phi)$.
   *
   * Polarity stores an orientation as two angles (theta = polar angle from the
   * z-axis, phi = azimuthal angle in the x-y plane) and inherits from a 2-element
   * vector. It is used in the framework to model cell polarisation.
   *
   * The struct provides conversion from/to Cartesian 3D vectors and several
   * force models:
   * - Polarisation forces (unidirectional / bidirectional alignment)
   * - Bending forces for tissue curvature
   * - Apical constriction forces
   * - Migration forces
   *
   * @tparam Scalar Floating-point type (e.g. `double`, `float`).
   * @tparam Align  Memory alignment (defaults to the scalar alignment).
   */
  template<typename Scalar, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) Polarity_ : public VectorN<Scalar, 2, Align> {
    using Base = VectorN<Scalar, 2, Align>;
    using Base::Base;

    /// Holds the resulting force vector and updated polarity from a force calculation.
    struct BendingForceResult {
      Vector3<Scalar> vector;  ///< The force vector (3D).
      Polarity_ polarity;      ///< The resulting polarity change.
    };

    /// @brief Default constructor (polarity at $(\theta=0, \phi=0)$).
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_() = default;

    /// @brief Construct with the same value for both $\theta$ and $\phi$.
    KOKKOS_INLINE_FUNCTION
    constexpr explicit Polarity_(Scalar value) : Base(value) { }

    /**
     * @brief Construct from spherical angles.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(Scalar theta, Scalar phi) : Base(theta, phi) { }

    /// @brief Inherit from the base vector type.
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Base& value) : Base(value) { }

    /**
     * @brief Construct polarity from a 3D displacement vector with a known distance.
     *
     * Derives $(\theta, \phi)$ from the Cartesian components.
     *
     * @param vector    3D displacement vector.
     * @param distance  Pre-computed length of the vector (avoids recomputation).
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Vector3<Scalar>& vector, const Scalar distance) : Base(
      Kokkos::acos(Kokkos::fmin(Scalar(1.0), Kokkos::fmax(Scalar(-1.0), vector[2] / distance))),
      Kokkos::atan2(vector[1], vector[0])
    ) { }

    /**
     * @brief Construct polarity from a 3D vector (computes the length internally).
     *
     * @param vector 3D Cartesian vector.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Vector3<Scalar>& vector) : Base(
      Kokkos::acos(Kokkos::fmin(Scalar(1.0), Kokkos::fmax(Scalar(-1.0), vector[2] / vector.length()))),
      Kokkos::atan2(vector[1], vector[0])
    ) { }

    /// @brief Access the polar angle $\theta$.
    KOKKOS_INLINE_FUNCTION constexpr Scalar& theta() { return this->data_[0]; }
    /// @brief Access the azimuthal angle $\phi$.
    KOKKOS_INLINE_FUNCTION constexpr Scalar& phi() { return this->data_[1]; }

    /// @copydoc theta()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& theta() const { return this->data_[0]; }
    /// @copydoc phi()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& phi() const { return this->data_[1]; }

    /**
     * @brief Create a Polarity_ from a 3D vector with a pre-computed distance.
     * @return The corresponding polar coordinates.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr Polarity_ from_vector3(const Vector3<Scalar>& vector, const Scalar distance) {
      return Polarity_(vector, distance);
    }

    /**
     * @brief Create a Polarity from a 3D vector (computes the length).
     * @return The corresponding polar coordinates.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr Polarity_ from_vector3(const Vector3<Scalar>& vector) {
      return Polarity_(vector);
    }

    /**
     * @brief Convert polar coordinates $(\theta, \phi)$ to a Cartesian 3D unit vector.
     * @return Unit vector $(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr Vector3<Scalar> to_vector3(const Polarity_& polarity) {
      return Vector3<Scalar>{
        Kokkos::sin(polarity[0]) * Kokkos::cos(polarity[1]),
        Kokkos::sin(polarity[0]) * Kokkos::sin(polarity[1]),
        Kokkos::cos(polarity[0])
      };
    }

    /// @copydoc to_vector3(const Polarity_&)
    KOKKOS_INLINE_FUNCTION
    constexpr Vector3<Scalar> to_vector3() const {
      return to_vector3(*this);
    }

    /**
     * @brief Dot product between two polarities (cosine of the angle between them).
     *
     * Computed as the dot product of the corresponding unit vectors in 3D.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar dot(const Polarity_& rhs) const {
      return Kokkos::sin(this->data_[0]) * Kokkos::sin(rhs[0]) *
        Kokkos::cos(this->data_[1] - rhs[1]) +
        Kokkos::cos(this->data_[0]) * Kokkos::cos(rhs[0]);
    }

    /**
     * @brief Force that aligns this polarity towards another polarity (one-sided).
     *
     * Returns the angular derivative (variation) of the dot product with respect
     * to this polarity's angles, i.e. the direction in $(\theta,\phi)$ space that
     * rotates this polarity toward the other.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_ unidirectional_polarization_force(const Polarity_& other) const {
      Polarity_ result{
        Kokkos::cos(this->data_[0]) * Kokkos::sin(other[0]) * Kokkos::cos(this->data_[1] - other[1]) - 
        Kokkos::sin(this->data_[0]) * Kokkos::cos(other[0]),
        0
      };

      Scalar sin_theta = Kokkos::sin(this->data_[0]);
      if (Kokkos::abs(sin_theta) > Kokkos::Experimental::epsilon_v<Scalar>)
        result[1] = -Kokkos::sin(other[0]) * Kokkos::sin(this->data_[1] - other[1]) / sin_theta;
      return result;
    }

    /**
     * @brief Force that aligns polarities towards each other (symmetric).
     *
     * Scales the unidirectional force by the current alignment (dot product),
     * so it naturally weakens as the polarities become parallel.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_ bidirectional_polarization_force(const Polarity_& other) const {
      return dot(other) * unidirectional_polarization_force(other);
    }

    /**
     * @brief Compute a bending force between two cells based on their polarities.
     *
     * Used to model active tissue bending. Returns both a 3D force vector and
     * an update to this polarity.
     *
     * @param displacement   Vector from this cell to the other.
     * @param other_polarity Polarity of the neighbouring cell.
     * @param distance       Distance from this cell to the other.
     * @return A BendingForceResult with the force vector and polarity change.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr BendingForceResult bending_force(
      const Vector3<Scalar>& displacement,
      const Polarity_& other_polarity,
      const Scalar distance
    ) const {
      Vector3<Scalar> this_vector = to_vector3();
      Scalar prod_i = this_vector.dot(displacement) / distance;

      Vector3<Scalar> other_vector = to_vector3(other_polarity);
      Scalar prod_j = other_vector.dot(displacement) / distance;

      return BendingForceResult{
        -prod_i / distance * this_vector + Kokkos::pow(prod_i, 2) / Kokkos::pow(distance, 2) * displacement +
        -prod_j / distance * other_vector + Kokkos::pow(prod_j, 2) / Kokkos::pow(distance, 2) * displacement,
        -prod_i * unidirectional_polarization_force(from_vector3(displacement, distance))
      };
    }

    /**
     * @brief Compute an apical constriction force between two cells.
     *
     * Similar to bending_force but incorporates a preferred angle that
     * drives the tissue toward a target curvature.
     *
     * @param displacement    Vector from this cell to the other.
     * @param distance        Distance from this cell to the other.
     * @param other_polarity  Polarity of the neighbouring cell.
     * @param preferred_angle Target angle for constriction.
     * @return A BendingForceResult with the force vector and polarity change.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr BendingForceResult apical_constriction_force(
      const Vector3<Scalar>& displacement,
      const Scalar distance,
      const Polarity_& other_polarity,
      const Scalar preferred_angle
    ) const {
      Vector3<Scalar> this_vector = to_vector3();
      Scalar prod_i = this_vector.dot(displacement) / distance + Kokkos::cos(preferred_angle);

      Vector3<Scalar> other_vector = to_vector3(other_polarity);
      Scalar prod_j = other_vector.dot(displacement) / distance - Kokkos::cos(preferred_angle);

      return BendingForceResult{
        -prod_i / distance * this_vector + Kokkos::pow(prod_i, 2) / Kokkos::pow(distance, 2) * displacement +
        -prod_j / distance * other_vector + Kokkos::pow(prod_j, 2) / Kokkos::pow(distance, 2) * displacement,
        -prod_i * unidirectional_polarization_force(from_vector3(displacement, distance))
      };
    }

    /**
     * @brief Compute a migration force that pulls a cell toward or pushes it
     *        away from another cell based on their polarities.
     *
     * The force switches direction depending on whether the neighbour is in
     * front of or behind the polarity direction.
     *
     * @param displacement   Vector from this cell to the other.
     * @param other_polarity Polarity of the neighbouring cell.
     * @param distance       Distance from this cell to the other.
     * @return 3D migration force vector.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Vector3<Scalar> migration_force(
      const Vector3<Scalar>& displacement,
      const Polarity_& other_polarity,
      const Scalar distance
    ) const {
      Vector3<Scalar> result{};
      Polarity_ displacement_polarity = from_vector3(displacement, distance);

      // pulling around other
      if (this->data_[0] != 0 || this->data_[1] != 0) {
        if (dot(displacement_polarity) <= -0.15) {
          Vector3<Scalar> this_vector = to_vector3();
          Vector3<Scalar> this_vector_T = displacement.orthonormal(this_vector);
          result = 0.6 * this_vector + 0.8 * this_vector_T;
        }
      }

      // pushed by other
      if (other_polarity[0] > Kokkos::Experimental::epsilon_v<Scalar> ||
          other_polarity[1] > Kokkos::Experimental::epsilon_v<Scalar>) {

        if (other_polarity.dot(displacement_polarity) >= 0.15) {
          Vector3<Scalar> other_vector = to_vector3(other_polarity);
          Vector3<Scalar> other_vector_T = (-displacement).orthonormal(other_vector);
          result -= 0.6 * other_vector + 0.8 * other_vector_T;
        }
      }

      return result;
    }
  };
} // namespace tecs



namespace HighFive::details {

  /**
   * @brief HighFive inspector that enables HDF5 read/write for Polarity.
   *
   * A Polarity is stored as a rank-1 dataset of two scalars
   * $(\theta, \phi)$, matching its underlying 2-element vector layout.
   */
  template <typename Scalar, unsigned int Align>
  struct inspector<tecs::Polarity_<Scalar, Align>> {
    using type = tecs::Polarity_<Scalar, Align>;
    using base_type = typename inspector<Scalar>::base_type;
    using hdf5_type = typename inspector<Scalar>::hdf5_type;

    static constexpr size_t ndim = inspector<Scalar>::ndim + 1;
    static constexpr size_t min_ndim = inspector<Scalar>::min_ndim + 1;
    static constexpr size_t max_ndim = inspector<Scalar>::max_ndim + 1;
    static constexpr bool is_trivially_nestable = inspector<Scalar>::is_trivially_nestable;
    static constexpr bool is_trivially_copyable =
      std::is_trivially_copyable<type>::value && inspector<Scalar>::is_trivially_copyable;

    /// @brief The rank is 1 plus the inner scalar's rank.
    static size_t getRank(const type& val) {
      return ndim + inspector<Scalar>::getRank(val[0]);
    }

    /// @brief Dimensions are `[2, ...]` — two angles plus any inner dimensions.
    static std::vector<size_t> getDimensions(const type& val) {
      std::vector<size_t> result = inspector<Scalar>::getDimensions(val[0]);
      result.insert(result.begin(), 2);
      return result;
    }

    /// @brief Prepare both components for reading.
    static void prepare(type& value, const std::vector<size_t>& next_dims) {
      for (unsigned int i = 0; i < 2; ++i)
        inspector<Scalar>::prepare(value[i], next_dims);
    }

    /// @brief Pointer to the first scalar component ($\theta$).
    static hdf5_type* data(type& value) {
      return inspector<Scalar>::data(value[0]);
    }

    /// @copydoc data(type&)
    static const hdf5_type* data(const type& value) {
      return inspector<Scalar>::data(value[0]);
    }
  };
} // namespace HighFive::details

#endif // TECS_TYPES_POLARITY_HPP
