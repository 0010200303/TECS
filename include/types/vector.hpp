#ifndef TECS_TYPES_VECTOR_HPP
#define TECS_TYPES_VECTOR_HPP

#include <type_traits>
#include <array>

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>
 
namespace tecs {

  /**
   * @brief A fixed-size, statically allocated N-dimensional vector.
   *
   * VectorN is the fundamental geometric type in the TECS framework. It stores
   * its components in a plain C array with configurable alignment, and works
   * seamlessly inside Kokkos parallel kernels. It provides:
   *
   * - Full set of component-wise arithmetic operators (+, -, *, /) for both
   *   vector-vector and vector-scalar operations.
   * - Standard vector math: dot product, length, distance, normalise, cross
   *   product (3D), and orthonormalisation.
   * - Iterator support for use with standard algorithms and range-for loops.
   * - Named accessors `.x()`, `.y()`, `.z()`, `.w()` for small dimensions.
   * - Template-index access via `.get<I>()`.
   *
   * Convenience aliases `Vector1`, `Vector2`, `Vector3`, `Vector4` are provided
   * for the most common dimensionalities.
   *
   * @tparam Scalar_      Floating-point type (e.g. `double`, `float`).
   * @tparam dimensions_  Number of components (must be > 0).
   * @tparam Align        Memory alignment (defaults to the scalar alignment).
   */
  template<typename Scalar_, unsigned int dimensions_, unsigned int Align = alignof(Scalar_)>
  struct alignas(Align) VectorN {
    using Scalar = Scalar_;
    /// Number of vector components.
    static constexpr unsigned int dimensions = dimensions_;

    static_assert(dimensions > 0, "Vector dimensions must be greater than 0");
    static_assert(std::is_trivially_copyable<Scalar>::value, "Scalar must be trivially copyable");

    using value_type = Scalar;
    using pointer = Scalar*;
    using const_pointer = const Scalar*;
    using reference = Scalar&;
    using const_reference = const Scalar&;
    using iterator = Scalar*;
    using const_iterator = const Scalar*;

    /// The raw component storage.
    Scalar data_[dimensions];

    /**
     * @brief Default constructor — initialises all components to zero.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN() : data_{} { }

    /**
     * @brief Construct with the same value for every component.
     * @param value Scalar value broadcast to all components.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr explicit VectorN(Scalar value) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] = value;
    }

    /**
     * @brief Construct from individual component values.
     *
     * The number of arguments must exactly match `dimensions`.
     * @code
     *   Vector3<double> v(1.0, 2.0, 3.0);
     * @endcode
     */
    template<typename... Args, std::enable_if_t<sizeof...(Args) == dimensions, int> = 0>
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN(Args... args) : data_{static_cast<Scalar>(args)...} { }

    /**
     * @brief Construct from a std::array.
     * @param array Source array of the same size.
     */
    constexpr explicit VectorN(const std::array<Scalar, dimensions>& array) {
      for (unsigned int i = 0; i < dimensions; ++i)
        data_[i] = array[i];
    }

    // --- STL container compatibility ---

    /// @brief Number of components (compile-time constant `dimensions`).
    KOKKOS_INLINE_FUNCTION constexpr std::size_t size() const { return dimensions; }
    /// @brief Pointer to the first component.
    KOKKOS_INLINE_FUNCTION constexpr Scalar* data() { return data_; }
    /// @copydoc data()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar* data() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr iterator begin() { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr iterator end() { return data_ + dimensions; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator begin() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator end() const { return data_ + dimensions; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator cbegin() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator cend() const { return data_ + dimensions; }

    // --- indexing ---

    /// @brief Access component by index (unchecked).
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& operator[](int i) { return data_[i]; }

    /// @copydoc operator[](int)
    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& operator[](int i) const { return data_[i]; }

    /// @brief Access component by compile-time index.
    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& get() {
      static_assert(I < dimensions, "Index out of bounds");
      return data_[I];
    }

    /// @copydoc get()
    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& get() const {
      static_assert(I < dimensions, "Index out of bounds");
      return data_[I];
    }

    // --- named accessors ---

    /// @brief First component.
    KOKKOS_INLINE_FUNCTION constexpr Scalar& x() { static_assert(dimensions > 0); return data_[0]; }
    /// @brief Second component (only if `dimensions > 1`).
    KOKKOS_INLINE_FUNCTION constexpr Scalar& y() { static_assert(dimensions > 1); return data_[1]; }
    /// @brief Third component (only if `dimensions > 2`).
    KOKKOS_INLINE_FUNCTION constexpr Scalar& z() { static_assert(dimensions > 2); return data_[2]; }
    /// @brief Fourth component (only if `dimensions > 3`).
    KOKKOS_INLINE_FUNCTION constexpr Scalar& w() { static_assert(dimensions > 3); return data_[3]; }

    /// @copydoc x()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& x() const { static_assert(dimensions > 0); return data_[0]; }
    /// @copydoc y()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& y() const { static_assert(dimensions > 1); return data_[1]; }
    /// @copydoc z()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& z() const { static_assert(dimensions > 2); return data_[2]; }
    /// @copydoc w()
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& w() const { static_assert(dimensions > 3); return data_[3]; }
  
    // --- vector-vector arithmetic operators ---

    /// @brief Component-wise addition.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator+=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] += rhs.data_[i];
      return *this;
    }

    /// @brief Component-wise subtraction.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] -= rhs.data_[i];
      return *this;
    }

    /// @brief Component-wise multiplication.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] *= rhs.data_[i];
      return *this;
    }

    /// @brief Component-wise division.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] /= rhs.data_[i];
      return *this;
    }

    /// @brief Component-wise addition.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(VectorN lhs, const VectorN& rhs) {
      lhs += rhs;
      return lhs;
    }

    /// @brief Component-wise subtraction.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(VectorN lhs, const VectorN& rhs) {
      lhs -= rhs;
      return lhs;
    }

    /// @brief Component-wise multiplication.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(VectorN lhs, const VectorN& rhs) {
      lhs *= rhs;
      return lhs;
    }

    /// @brief Component-wise division.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(VectorN lhs, const VectorN& rhs) {
      lhs /= rhs;
      return lhs;
    }

    // --- vector-scalar arithmetic operators ---

    /// @brief Add scalar to every component.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator+=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] += rhs;
      return *this;
    }

    /// @brief Subtract scalar from every component.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] -= rhs;
      return *this;
    }

    /// @brief Multiply every component by scalar.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] *= rhs;
      return *this;
    }

    /// @brief Divide every component by scalar.
    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] /= rhs;
      return *this;
    }

    /// @brief Vector + scalar (scalar added to each component).
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(VectorN lhs, Scalar rhs) {
      lhs += rhs;
      return lhs;
    }

    /// @brief Vector - scalar.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(VectorN lhs, Scalar rhs) {
      lhs -= rhs;
      return lhs;
    }

    /// @brief Vector * scalar.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(VectorN lhs, Scalar rhs) {
      lhs *= rhs;
      return lhs;
    }

    /// @brief Vector / scalar.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(VectorN lhs, Scalar rhs) {
      lhs /= rhs;
      return lhs;
    }

    /// @brief Scalar + vector.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(Scalar lhs, VectorN rhs) {
      rhs += lhs;
      return rhs;
    }

    /// @brief Scalar - vector (subtract each component from the scalar).
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(Scalar lhs, VectorN rhs) {
      for (int i = 0; i < dimensions; ++i)
        rhs.data_[i] = lhs - rhs.data_[i];
      return rhs;
    }

    /// @brief Scalar * vector.
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(Scalar lhs, VectorN rhs) {
      rhs *= lhs;
      return rhs;
    }

    /// @brief Scalar / vector (divide scalar by each component).
    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(Scalar lhs, VectorN rhs) {
      for (int i = 0; i < dimensions; ++i)
        rhs.data_[i] = lhs / rhs.data_[i];
      return rhs;
    }

    // --- vector math ---

    /**
     * @brief Dot product with another vector.
     * @return $\sum_{i=0}^{d-1} a_i b_i$.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar dot(const VectorN& rhs) const {
      Scalar result{};
      for (int i = 0; i < dimensions; ++i)
        result += data_[i] * rhs.data_[i];
      return result;
    }

    /// @brief Squared Euclidean norm $|\mathbf{v}|^2$.
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar length_squared() const {
      return dot(*this);
    }

    /// @brief Euclidean norm $|\mathbf{v}|$.
    KOKKOS_INLINE_FUNCTION
    Scalar length() const {
      return Kokkos::sqrt(length_squared());
    }

    /**
     * @brief Squared Euclidean distance to another vector.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar distance_to_squared(const VectorN& rhs) const {
      Scalar result{};
      for (int i = 0; i < dimensions; ++i) {
        const Scalar d = data_[i] - rhs.data_[i];
        result += d * d;
      }
      return result;
    }

    /**
     * @brief Euclidean distance to another vector.
     */
    KOKKOS_INLINE_FUNCTION
    Scalar distance_to(const VectorN& rhs) const {
      return Kokkos::sqrt(distance_to_squared(rhs));
    }

    /// @brief Normalise in-place (divide by length) and return self.
    KOKKOS_INLINE_FUNCTION
    VectorN& normalize() {
      const Scalar inv_norm = Scalar(1) / length();
      for (int i = 0; i < dimensions; ++i)
        data_[i] *= inv_norm;
      return *this;
    }

    /// @brief Return a normalised copy (does not modify this vector).
    KOKKOS_INLINE_FUNCTION
    VectorN normalized() const {
      VectorN result = *this;
      result.normalize();
      return result;
    }

    /**
     * @brief Cross product (3D only).
     * @return $\mathbf{a} \times \mathbf{b}$.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN cross(const VectorN& rhs) const {
      static_assert(dimensions == 3, "Cross product is only defined for 3D vectors");
      return VectorN{
        data_[1] * rhs.data_[2] - data_[2] * rhs.data_[1],
        data_[2] * rhs.data_[0] - data_[0] * rhs.data_[2],
        data_[0] * rhs.data_[1] - data_[1] * rhs.data_[0]
      };
    }

    /**
     * @brief Gram–Schmidt orthonormalisation: project this vector onto the
     *        subspace orthogonal to @p rhs, then normalise.
     *
     * @f[
     *   \mathbf{v}_\perp = \frac{\mathbf{v} - (\mathbf{v}\cdot\mathbf{u})\mathbf{u}}{|\mathbf{v}
     *   - (\mathbf{v}\cdot\mathbf{u})\mathbf{u}|}
     * @f]
     *
     * @param rhs Vector to orthogonalise against (must be unit length).
     * @return Orthonormalised vector.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN orthonormal(const VectorN& rhs) const {
      VectorN normal = *this - dot(rhs) * rhs;
      return normal / Kokkos::sqrt(normal.dot(normal));
    }

    /**
     * @brief Clockwise perpendicular in 2D: $(y, -x)$.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN perpendicular_cw() const {
      static_assert(dimensions == 2, "perpendicular is only defined for 2D vectors");
      return VectorN{data_[1], -data_[0]};
    }

    /**
     * @brief Counter-clockwise perpendicular in 2D: $(-y, x)$.
     */
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN perpendicular_ccw() const {
      static_assert(dimensions == 2, "perpendicular is only defined for 2D vectors");
      return VectorN{-data_[1], data_[0]};
    }

    // --- unary operators ---

    /// @brief Unary plus (no-op).
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN operator+() const {
      return *this;
    }

    /// @brief Unary negation — negate every component.
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN operator-() const {
      VectorN result;
      for (int i = 0; i < dimensions; ++i)
        result.data_[i] = -data_[i];
      return result;
    }
  };

  /// @brief 1-component vector.
  template<typename Scalar>
  using Vector1 = VectorN<Scalar, 1>;

  /// @brief 2-component vector.
  template<typename Scalar>
  using Vector2 = VectorN<Scalar, 2>;

  /// @brief 3-component vector.
  template<typename Scalar>
  using Vector3 = VectorN<Scalar, 3>;

  /// @brief 4-component vector.
  template<typename Scalar>
  using Vector4 = VectorN<Scalar, 4>;
} // namespace tecs



/**
 * @brief Kokkos reduction identity for tecs::VectorN (zero vector as identity).
 *
 * This enables the use of VectorN as a reduction target in Kokkos parallel
 * operations (e.g. `Kokkos::parallel_reduce`).
 */
namespace Kokkos {
  template<typename Scalar, unsigned int dimensions, unsigned int Align>
  struct reduction_identity<tecs::VectorN<Scalar, dimensions, Align>> {
    using value_type = tecs::VectorN<Scalar, dimensions, Align>;

    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr value_type sum() {
      return value_type{};
    }
  };
} // namespace Kokkos

/**
 * @brief HighFive inspector that enables HDF5 read/write for VectorN.
 *
 * A VectorN is stored as a rank-1 dataset of `dimensions` scalars,
 * matching its in-memory layout. Nested vectors (e.g. `VectorN<VectorN<...>>`)
 * are handled recursively through the inner inspector.
 */
namespace HighFive::details {
  template <typename Scalar, unsigned int dimensions, unsigned int Align>
  struct inspector<tecs::VectorN<Scalar, dimensions, Align>> {
    using type = tecs::VectorN<Scalar, dimensions, Align>;
    using base_type = typename inspector<Scalar>::base_type;
    using hdf5_type = typename inspector<Scalar>::hdf5_type;

    static constexpr size_t ndim = inspector<Scalar>::ndim + 1;
    static constexpr size_t min_ndim = inspector<Scalar>::min_ndim + 1;
    static constexpr size_t max_ndim = inspector<Scalar>::max_ndim + 1;
    static constexpr bool is_trivially_nestable = inspector<Scalar>::is_trivially_nestable;
    static constexpr bool is_trivially_copyable =
      std::is_trivially_copyable<type>::value && inspector<Scalar>::is_trivially_copyable;

    /// @brief Rank is 1 plus the inner scalar's rank.
    static size_t getRank(const type& val) {
      return ndim + inspector<Scalar>::getRank(val[0]);
    }

    /// @brief Dimensions are `[dimensions, ...]` - N components plus any inner dimensions.
    static std::vector<size_t> getDimensions(const type& val) {
      std::vector<size_t> result = inspector<Scalar>::getDimensions(val[0]);
      result.insert(result.begin(), dimensions);
      return result;
    }

    /// @brief Prepare each component for reading.
    static void prepare(type& value, const std::vector<size_t>& next_dims) {
      for (unsigned int i = 0; i < dimensions; ++i)
        inspector<Scalar>::prepare(value[i], next_dims);
    }

    /// @brief Pointer to the first scalar component.
    static hdf5_type* data(type& value) {
      return inspector<Scalar>::data(value[0]);
    }

    /// @copydoc data(type&)
    static const hdf5_type* data(const type& value) {
      return inspector<Scalar>::data(value[0]);
    }

    /// @brief Flatten into a contiguous HDF5 buffer, one component at a time.
    static void serialize(const type& val, const std::vector<size_t>& m_dims, hdf5_type* m) {
      const size_t count = (m_dims.empty() ? 1 : m_dims[0]);
      for (size_t i = 0; i < count; ++i) {
        std::vector<size_t> inner_dims(m_dims.begin() + 1, m_dims.end());
        inspector<Scalar>::serialize(val.data_[i], inner_dims, m + i);
      }
    }

    /// @brief Reconstruct from a contiguous HDF5 buffer.
    static void unserialize(const hdf5_type* vec, const std::vector<size_t>& m_dims, type& val) {
      const size_t count = (m_dims.empty() ? 1 : m_dims[0]);
      for (size_t i = 0; i < count; ++i) {
        std::vector<size_t> inner_dims(m_dims.begin() + 1, m_dims.end());
        inspector<Scalar>::unserialize(vec + i, inner_dims, val.data_[i]);
      }
    }
  };
} // namespace HighFive::details

#endif // TECS_TYPES_VECTOR_HPP
