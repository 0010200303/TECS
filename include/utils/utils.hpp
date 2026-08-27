#ifndef TECS_UTILS_UTILS_HPP
#define TECS_UTILS_UTILS_HPP

#include <array>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "types/vector.hpp"
#include "types/polarity.hpp"
#include "types/link.hpp"

namespace tecs {

  /**
   * @brief Extract a homogeneous `std::array` type from a tuple of field specs.
   *
   * All fields must share the same underlying value type (enforced by static_assert).
   *
   * @tparam Tuple A `std::tuple<Spec...>` where each `Spec` has a `type` member.
   */
  template<typename Tuple>
  struct extract_types;

  /// @copydoc extract_types
  template<typename... Fs>
  struct extract_types<std::tuple<Fs...>> {
    static_assert(sizeof...(Fs) > 0, "Storage requires at least one field");
    using first_type = typename std::tuple_element_t<0, std::tuple<Fs...>>::type;
    static_assert((std::is_same_v<first_type, typename Fs::type> && ...),
      "All Storage field types must match");

    using type = std::array<first_type, sizeof...(Fs)>;
  };

  /// @brief Shorthand for `extract_types<Tuple>::type`.
  template<typename Tuple>
  using storage_t = typename extract_types<Tuple>::type;

  /**
   * @brief A fixed-size bundle of field values (one per field spec), usable
   *        inside Kokkos kernels.
   *
   * `ValuesFromFields<Fields>::type` is a struct with a `data[]` array holding
   * one value per field. It supports `+=` for accumulating multiple contributions
   * (e.g. from different forces).
   *
   * @tparam Fields A `std::tuple<Spec...>` where each `Spec` defines a field.
   */
  template<typename Fields>
  struct ValuesFromFields;

  template<typename... Specs>
  struct ValuesFromFields<std::tuple<Specs...>> {
    struct type {
      public:
        static_assert(sizeof...(Specs) > 0, "ValuesFromFields requires at least one field");

        using value_type = typename std::tuple_element_t<0, std::tuple<Specs...>>::type::value_type;
        static_assert((std::is_same_v<value_type, typename Specs::type::value_type> && ...),
          "All field value types must match");

        using tuple_type = value_type[sizeof...(Specs)];

        /// The per-field values, accessible as `data[0]`, `data[1]`, …
        tuple_type data{};

        KOKKOS_INLINE_FUNCTION
        type() = default;

        /**
         * @brief Construct from per-field values (one argument per field).
         */
        template<typename... Args, std::enable_if_t<sizeof...(Args) == sizeof...(Specs), int> = 0>
        KOKKOS_INLINE_FUNCTION
        explicit type(Args&&... args) : data{static_cast<value_type>(std::forward<Args>(args))...} { }

        /// @brief Component-wise addition (fold over all fields).
        KOKKOS_INLINE_FUNCTION
        type& operator+=(const type& rhs) {
          add_impl(rhs, std::make_index_sequence<sizeof...(Specs)>{});
          return *this;
        }

    private:
      template<std::size_t... I>
      KOKKOS_INLINE_FUNCTION
      void add_impl(const type& rhs, std::index_sequence<I...>) {
        ((data[I] += rhs.data[I]), ...);
      }
    };
  };

  // --- SFINAE helpers for read/write/access dispatch ---

  template<typename T, typename = void>
  struct has_read : std::false_type {};

  template<typename T>
  struct has_read<T, std::void_t<decltype(std::declval<const T&>().read(0))>> : std::true_type {};

  template<typename T, typename = void>
  struct has_write : std::false_type {};

  template<typename T>
  struct has_write<T, std::void_t<decltype(std::declval<T&>().write(0))>> : std::true_type {};

  template<typename T, typename = void>
  struct has_access : std::false_type {};

  template<typename T>
  struct has_access<T, std::void_t<decltype(std::declval<T&>().access(0))>> : std::true_type {};

  /**
   * @brief Read an element from a View using its `read()` method if available,
   *        otherwise falls back to `operator()`.
   *
   * This lets generic code work with both tecs::View (which has explicit
   * read/write/access) and plain Kokkos::View (which only has operator()).
   */
  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto read_element(const ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_read<ViewType>::value)
      return v.read(i);
    else
      return v(i);
  }

  /**
   * @brief Write an element into a View using its `write()` method if available.
   * @copydetails read_element
   */
  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto write_element(ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_write<ViewType>::value)
      return v.write(i);
    else
      return v(i);
  }

  /**
   * @brief Access an element without setting modified flags.
   * @copydetails read_element
   */
  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto access_element(ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_access<ViewType>::value)
      return v.access(i);
    else
      return v(i);
  }
} // namespace tecs

// =========================================================================
// Force definition macros - make defining force functors more concise.
//
// Each force type has two macro families:
//
// **Lambda-style** (inline usage inside `sim.take_step()`):
//   `GENERIC_FORCE(...)`  / `PAIRWISE_FORCE(...)`  / `LINK_FORCE(...)`  / `UPDATE_FUNC(...)`
//
// **Functor-style** (reusable named struct with `operator()`):
//   `GENERIC_FORCE_OP()`  / `PAIRWISE_FORCE_OP()`  / `LINK_FORCE_OP()`  / `UPDATE_FUNC_OP()`
//
// See the individual macro docs below for the available parameters.
// =========================================================================

/// @brief Declare a field reference variable of generic-force type.
#define GENERIC_REF(__TYPE__, __NAME__) detail::GenericFieldRef<__TYPE__> __NAME__
/// @brief Declare a field reference variable of pairwise-force type.
#define PAIRWISE_REF(__TYPE__, __NAME__) detail::PairwiseFieldRef<__TYPE__> __NAME__

// --- Implementation helpers (not intended for direct use) ---
#define GENERIC_FORCE_IMPL tecs::detail::generic_force | KOKKOS_LAMBDA
#define PAIRWISE_FORCE_IMPL tecs::detail::pairwise_force | KOKKOS_LAMBDA
#define UPDATE_FUNC_IMPL tecs::detail::update_func | KOKKOS_LAMBDA
#define LINK_FORCE_IMPL tecs::detail::link_force | KOKKOS_LAMBDA

/**
 * @brief Parameters passed to a generic (per-agent) force lambda.
 *
 * - `is_full_step` - `true` for the main evaluation, `false` for sub-steps.
 * - `i`            - agent index.
 * - `rng`          - Kokkos random generator state.
 * - `ctx`          - field references (the `GENERIC_REF` variables).
 */
#define GENERIC_FORCE_PARAMETERS const bool is_full_step, const unsigned int i, \
  Random& rng, const GenericForceFields& ctx

/**
 * @brief Parameters passed to a pairwise force lambda.
 *
 * - `is_full_step` - `true` only once over all integration stages (useful for avoiding double counting).
 * - `i, j`         - indices of the two interacting agents.
 * - `displacement` - vector from i to j.
 * - `distance`     - length of the displacement vector.
 * - `rng`          - Kokkos random generator state.
 * - `drag`         - drag coefficient (can be modified).
 * - `ctx`          - field references (the `PAIRWISE_REF` variables).
 */
#define PAIRWISE_FORCE_PARAMETERS const bool is_full_step, const unsigned int i, const unsigned int j, \
  const Vector& displacement, const Scalar distance, Random& rng, Scalar& drag, const PairwiseForceFields& ctx

/**
 * @brief Parameters passed to an update function.
 *
 * - `i`   - agent index.
 * - `rng` - Kokkos random generator state.
 */
#define UPDATE_FUNC_PARAMETERS const unsigned int i, Random& rng

/**
 * @brief Parameters passed to a link force lambda.
 *
 * - `is_full_step` - `true` only once over all integration stages (useful for avoiding double counting).
 * - `link`         - the link (a, b) between two agents.
 * - `rng`          - Kokkos random generator state.
 * - `ctx`          - field references (the `GENERIC_REF`/`PAIRWISE_REF` variables).
 */
#define LINK_FORCE_PARAMETERS const bool is_full_step, const Link& link, Random& rng, const LinkForceFields& ctx

/**
 * @brief Inline generic force lambda.
 * @code
 *   GENERIC_FORCE({
 *     ctx.position(i) += 1.0;
 *   })
 * @endcode
 */
#define GENERIC_FORCE(...) [&]() { return GENERIC_FORCE_IMPL(GENERIC_FORCE_PARAMETERS) { __VA_ARGS__ }; }

/**
 * @brief Inline pairwise force lambda.
 * @copydetails GENERIC_FORCE
 */
#define PAIRWISE_FORCE(...) [&]() { return PAIRWISE_FORCE_IMPL(PAIRWISE_FORCE_PARAMETERS) { __VA_ARGS__ }; }

/**
 * @brief Inline update function lambda.
 * @copydetails GENERIC_FORCE
 */
#define UPDATE_FUNC(...) [&]() { return UPDATE_FUNC_IMPL(UPDATE_FUNC_PARAMETERS) { __VA_ARGS__ }; }

/**
 * @brief Inline link force lambda.
 * @copydetails GENERIC_FORCE
 */
#define LINK_FORCE(...) [&]() { return LINK_FORCE_IMPL(LINK_FORCE_PARAMETERS) { __VA_ARGS__ }; }

/**
 * @brief Functor-style generic force operator.
 *
 * Use inside a named struct to define a reusable force:
 * @code
 *   struct MyForce {
 *     GENERIC_FORCE_OP() {
 *       ctx.position(i) += 1.0;
 *     }
 *   };
 * @endcode
 */
#define GENERIC_FORCE_OP \
  using tag = tecs::detail::GenericForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(GENERIC_FORCE_PARAMETERS) const

/// @brief Functor-style pairwise force operator.
#define PAIRWISE_FORCE_OP \
  using tag = tecs::detail::PairwiseForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(PAIRWISE_FORCE_PARAMETERS) const

/// @brief Functor-style update function operator.
#define UPDATE_FUNC_OP \
  using tag = tecs::detail::UpdateFuncTag; \
  KOKKOS_INLINE_FUNCTION void operator()(UPDATE_FUNC_PARAMETERS) const

/// @brief Functor-style link force operator.
#define LINK_FORCE_OP \
  using tag = tecs::detail::LinkForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(LINK_FORCE_PARAMETERS) const

/**
 * @brief Inline initialization lambda (for use in init callbacks).
 * @code
 *   INIT_FUNC({
 *     ctx.position(i) = rng.rand(0.0, 1.0);
 *   })
 * @endcode
 */
#define INIT_FUNC(...) [&]() { return KOKKOS_LAMBDA(const unsigned int i, Random& rng) { __VA_ARGS__ }; }

/**
 * @brief Functor-style initilization operator.
 * @code
 *   INIT_OP{
 *     positions_view(i) = Vector(Scalar(i) * distance);
 *   }
 * @endcode
 */
#define INIT_OP KOKKOS_INLINE_FUNCTION void operator()(const unsigned int i, Random& generator) const

// =========================================================================
// Type extraction macros - bring simulation config types into scope.
// =========================================================================

/**
 * @brief Extract core types from a SimulationConfig into the current scope.
 *
 * Defines: `Scalar`, `dimensions`, `Vector`, `VectorI`, `VectorView`,
 * `Polarity`, `Plane`, `RandomPool`, `Random`, `GenericForceFields`,
 * `PairwiseForceFields`, `LinkForceFields`.
 */
#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = tecs::VectorN<Scalar, dimensions>; \
  using VectorI = tecs::VectorN<int, dimensions>; \
  using VectorView = tecs::View<Vector>; \
  using Polarity = tecs::Polarity_<Scalar>; \
  using Plane = tecs::PlaneN<Scalar, dimensions>; \
  using RandomPool = typename __SIMULATION_CONFIG__::RandomPoolT; \
  using Random = typename RandomPool::generator_type; \
  using GenericForceFields = typename __SIMULATION_CONFIG__::template ForceFields<tecs::detail::GenericFieldRef>; \
  using PairwiseForceFields = typename __SIMULATION_CONFIG__::template ForceFields<tecs::detail::PairwiseFieldRef>; \
  using LinkForceFields = typename __SIMULATION_CONFIG__::template ForceFields<tecs::detail::LinkFieldRef>;

/**
 * @brief Like `EXTRACT_TYPES_FROM_SIMULATION_CONFIG` plus component types.
 *
 * Additionally defines: `Fields`, `Integrator`, `PairFinder`, `ComFixer`, `Writer`.
 */
#define EXTRACT_ALL_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Fields = typename __SIMULATION_CONFIG__::Fields; \
  using Integrator = typename __SIMULATION_CONFIG__::template IntegratorT<__SIMULATION_CONFIG__>; \
  using PairFinder = typename __SIMULATION_CONFIG__::template PairFinderT<__SIMULATION_CONFIG__>; \
  using ComFixer = typename __SIMULATION_CONFIG__::template ComFixerT<__SIMULATION_CONFIG__>; \
  using Writer = typename __SIMULATION_CONFIG__::template WriterT<__SIMULATION_CONFIG__>;

/**
 * @brief Extract `Scalar` and `dimensions` from a vector type.
 *
 * Used inside generic code that receives a vector template parameter.
 */
#define EXTRACT_VECTOR(__VECTOR__) \
  using Scalar = typename __VECTOR__::Scalar; \
  static constexpr unsigned int dimensions = __VECTOR__::dimensions; \

#endif // TECS_UTILS_UTILS_HPP
