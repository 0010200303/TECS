#ifndef TECS_SIMULATION_CONFIG
#define TECS_SIMULATION_CONFIG

#include <string_view>

#include <Kokkos_Random.hpp>

#include "types/view.hpp"
#include "integrators/detail.hpp"
#include "integrators/heun.hpp"
#include "pair_finders/naive_all_pairs.hpp"
#include "com_fixers/com_fixers.hpp"
#include "io/hdf5_writer.hpp"

namespace tecs {
  namespace detail {

    /**
     * @brief A compile-time fixed-size string used as a non-type template parameter.
     *
     * Used by `Field` to embed the field name as part of the type.
     */
    template <std::size_t N>
    struct fixed_string {
      char data[N];

      consteval fixed_string(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i)
          data[i] = str[i];
      }

      constexpr operator std::string_view() const {
        return {data, N - 1};
      }
    };

    /**
     * @brief Associates a C++ type with a compile-time name - one field in the
     *        simulation's data layout.
     *
     * @tparam T    The value type (e.g. `Vector`, `Scalar`).
     * @tparam Name The field name as a `fixed_string` (e.g. `"position"`).
     */
    template<typename T, fixed_string Name>
    struct Field {
      using type = T;
      static constexpr auto name = Name;
    };

    /// @brief A typelist of Fields - defines the complete internal simulation data layout.
    template <typename... Fields>
    struct FieldList {};

    /// @brief Count pointer indirections (0 = value, 1 = pointer).
    template<typename U>
    struct pointer_depth { static constexpr std::size_t value = 0; };
    template<typename U>
    struct pointer_depth<U*> { static constexpr std::size_t value = 1 + pointer_depth<U>::value; };

    /**
     * @brief Maps a Field to its tecs::View type (host-syncable).
     * Pointers in the field type (e.g. `Vector*`) become `View<Vector>`.
     */
    template <typename Field>
    struct ViewFromField {
      using field_type = std::remove_cv_t<typename Field::type>;
      static_assert(pointer_depth<field_type>::value <= 1,
                    "Field element types must not be pointer-to-pointer or higher (depth >= 2)");
      using type = View<std::remove_pointer_t<field_type>>;
    };

    /**
     * @brief Maps a Field to a device-only Kokkos::View (no host sync).
     * Used for internal integrator / pair-finder buffers.
     */
    template <typename Field>
    struct DeviceViewFromField {
      using field_type = std::remove_cv_t<typename Field::type>;
      using type = Kokkos::View<std::remove_pointer_t<field_type>*>;
    };

    /// @brief Create a View for a field with a given size @p n.
    template <typename Field>
    auto make_view(std::size_t n) {
      using view_type = typename ViewFromField<Field>::type;
      return view_type(std::string(Field::name), n);
    }

    /// @brief Extract the first Field from a FieldList.
    template <typename FieldList>
    struct FirstFieldFromList;

    template <typename Field, typename... Rest>
    struct FirstFieldFromList<FieldList<Field, Rest...>> {
      using type = Field;
    };

    /// @brief Build the PairFinder type from the config's fields.
    template <template<typename, typename, int> typename PairFinderT, typename SimulationConfig, typename FieldList>
    struct PairFinderFromFields;

    template <template<typename, typename, int> typename PairFinderT, typename SimulationConfig, typename... Fields>
    struct PairFinderFromFields<PairFinderT, SimulationConfig, FieldList<Fields...>> {
      using FirstField = typename FirstFieldFromList<FieldList<Fields...>>::type;
      using type = PairFinderT<
        typename DeviceViewFromField<FirstField>::type,
        typename SimulationConfig::Scalar, SimulationConfig::dimensions
      >;
    };

    /// @brief Build the Integrator type from the config's fields.
    template <template<typename, typename...> typename IntegratorT, typename SimulationConfig, typename FieldList>
    struct IntegratorFromFields;

    template <template<typename, typename...> typename IntegratorT, typename SimulationConfig, typename... Fields>
    struct IntegratorFromFields<IntegratorT, SimulationConfig, FieldList<Fields...>> {
      // Integrators receive device only Kokkos::Views for internal storage
      // (The simulation's storage retains tecs::View for host sync management)
      using type = IntegratorT<
        typename SimulationConfig::template PairFinderT<SimulationConfig>,
        typename SimulationConfig::template ComFixerT<SimulationConfig>,
        typename SimulationConfig::template ForceFields<tecs::detail::GenericFieldRef>,
        typename SimulationConfig::template ForceFields<tecs::detail::PairwiseFieldRef>,
        typename SimulationConfig::template ForceFields<tecs::detail::LinkFieldRef>,
        typename DeviceViewFromField<Fields>::type...
      >;
    };

    /// @brief Alias helper: resolve the integrator type from config.
    template<template<typename, typename...> typename IntegratorT, typename SimulationConfig>
    using integrator_t = typename detail::IntegratorFromFields<IntegratorT, SimulationConfig, typename SimulationConfig::Fields>::type;

    /// @brief Alias helper: resolve the pair-finder type from config.
    template<template<typename, typename, int> typename PairFinderT, typename SimulationConfig>
    using pair_finder_t = typename detail::PairFinderFromFields<PairFinderT, SimulationConfig, typename SimulationConfig::Fields>::type;

    /// @brief Alias helper: resolve the writer type from config.
    template<template<typename> typename WriterT, typename SimulationConfig>
    using writer_t = WriterT<SimulationConfig>;

    /// @brief Holds a single field View (used by FieldStorage).
    template <typename Field>
    struct FieldHolder {
      using view_type = typename ViewFromField<Field>::type;
      view_type view;

      FieldHolder(view_type v) : view(v) { }
    };

    /// @brief Storage for all simulation fields, one View per field.
    template <typename... Fields>
    struct FieldStorage : FieldHolder<Fields>... {
      FieldStorage(std::size_t n) : FieldHolder<Fields>{make_view<Fields>(n)}... { }
    };

    /// @brief Resolve FieldStorage from a FieldList.
    template <typename FieldList>
    struct FieldStorageFromList;

    template <typename... Fields>
    struct FieldStorageFromList<detail::FieldList<Fields...>> {
      using type = FieldStorage<Fields...>;
    };

    /// @brief Retrieve the View for a given Field from a FieldStorage.
    template <typename Field, typename Storage>
    inline auto& get(Storage& s) {
      using holder = FieldHolder<Field>;
      return static_cast<holder&>(s).view;
    }

    /// @copydoc get
    template <typename Field, typename Storage>
    inline const auto& get(const Storage& s) {
      using holder = FieldHolder<Field>;
      return static_cast<const holder&>(s).view;
    }

    /// @brief Extract a tuple of all Views from a FieldStorage.
    template <typename FieldList, typename Storage>
    struct ViewsFromStorage;

    template <typename... Fields, typename Storage>
    struct ViewsFromStorage<FieldList<Fields...>, Storage> {
      static auto get(Storage& storage) {
        return std::forward_as_tuple(detail::get<Fields>(storage)...);
      }

      static auto get(const Storage& storage) {
        return std::forward_as_tuple(detail::get<Fields>(storage)...);
      }
    };
  } // namespace detail

// =========================================================================
// Configuration macros - used inside a user's SimulationConfig struct.
//
// Example:
// @code
//   struct MyConfig : public DefaultSimulationConfig {
//     CONFIG_INTEGRATOR(integrators::Euler)
//     CONFIG_PAIR_FINDER(pair_finders::NaiveAllPairs)
//     CONFIG_FIELDS(
//       (Vector, position),
//       (Vector, velocity)
//     )
//   };
// @endcode
// =========================================================================

/// @brief Set the floating-point scalar type.
#define CONFIG_SCALAR(__SCALAR__) \
  using Scalar = __SCALAR__;

/// @brief Set the number of spatial dimensions.
#define CONFIG_DIMENSIONS(__DIMENSIONS__) \
  static constexpr int dimensions = __DIMENSIONS__;

/// @brief Set the Kokkos random number pool type.
#define CONFIG_RANDOM_POOL(__RANDOM_POOL__) \
  using RandomPoolT = __RANDOM_POOL__<>;

/// @brief Set the integrator (e.g. `integrators::Euler`, `integrators::Heun`).
#define CONFIG_INTEGRATOR(__INTEGRATOR__) \
  template<typename SimulationConfig> \
  using IntegratorT = tecs::detail::integrator_t<__INTEGRATOR__, SimulationConfig>;

/// @brief Set the pair-finder algorithm (e.g. `pair_finders::NaiveAllPairs`).
#define CONFIG_PAIR_FINDER(__PAIR_FINDER__) \
  template<typename SimulationConfig> \
  using PairFinderT = tecs::detail::pair_finder_t<__PAIR_FINDER__, SimulationConfig>;

/// @brief Set the centre-of-mass fixer.
#define CONFIG_COM_FIXER(__COM_FIXER__) \
  template<typename SimulationConfig> \
  using ComFixerT = __COM_FIXER__<SimulationConfig>;

/// @brief Set the output writer (e.g. `io::HDF5_Writer`, `io::Dummy`).
#define CONFIG_WRITER(__WRITER__) \
  template<typename SimulationConfig> \
  using WriterT = tecs::detail::writer_t<__WRITER__, SimulationConfig>;

// --- Internal preprocessor helpers for CONFIG_FIELDS ---

#define FIELD(__SCALAR_TYPE__, __FIELD_NAME__) \
  tecs::detail::Field<__SCALAR_TYPE__, #__FIELD_NAME__>

#define APPLY_PAIR(M, PAIR) M PAIR

// PP_NARG: count the number of arguments
#define PP_NARG(...) PP_NARG_IMPL(__VA_ARGS__,8,7,6,5,4,3,2,1,0)
#define PP_NARG_IMPL(x8,x7,x6,x5,x4,x3,x2,x1,n,...) n

// CAT with argument expansion (avoids ## not expanding args)
#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a ## b

// FOR_EACH_PAIR: iterate over (type, name) pairs (comma-separated for FieldList)
#define FOR_EACH_PAIR_1(M, A) APPLY_PAIR(M, A)
#define FOR_EACH_PAIR_2(M, A, B) APPLY_PAIR(M, A), APPLY_PAIR(M, B)
#define FOR_EACH_PAIR_3(M, A, B, C) APPLY_PAIR(M, A), APPLY_PAIR(M, B), APPLY_PAIR(M, C)
#define FOR_EACH_PAIR_4(M, A, B, C, D) APPLY_PAIR(M, A), APPLY_PAIR(M, B), APPLY_PAIR(M, C), APPLY_PAIR(M, D)

#define FOR_EACH_PAIR(M, ...) CAT(FOR_EACH_PAIR_, PP_NARG(__VA_ARGS__))(M, __VA_ARGS__)

// helpers for generating ForceFields struct members (semicolon-separated)
#define FIELDS_ITERATE_1(M, A) APPLY_PAIR(M, A);
#define FIELDS_ITERATE_2(M, A, B) APPLY_PAIR(M, A); APPLY_PAIR(M, B);
#define FIELDS_ITERATE_3(M, A, B, C) APPLY_PAIR(M, A); APPLY_PAIR(M, B); APPLY_PAIR(M, C);
#define FIELDS_ITERATE_4(M, A, B, C, D) APPLY_PAIR(M, A); APPLY_PAIR(M, B); APPLY_PAIR(M, C); APPLY_PAIR(M, D);

#define FIELDS_ITERATE(M, ...) CAT(FIELDS_ITERATE_, PP_NARG(__VA_ARGS__))(M, __VA_ARGS__)

#define FIELD_FORCE_MEMBER(TYPE, NAME) FieldRefT<TYPE> NAME

// --- Field type resolution helpers (deferred evaluation) ---
// Maps a type name token like `Vector` to its template form.
#define TECS_FIELD_TYPE(T, S, D) CAT(TECS_FIELD_, T)(S, D)
#define TECS_FIELD_Vector(S, D) tecs::VectorN<S, D>
#define TECS_FIELD_Polarity(S, D) tecs::Polarity_<S>
#define TECS_FIELD_Scalar(S, D) S
#define TECS_FIELD_int(S, D) int
#define TECS_FIELD_unsigned(S, D) unsigned int
#define TECS_FIELD_long(S, D) long
#define TECS_FIELD_float(S, D) float
#define TECS_FIELD_double(S, D) double
#define TECS_FIELD_bool(S, D) bool

#define TECS_FIELD_TPL(T, N) tecs::detail::Field<TECS_FIELD_TYPE(T, __TECS_SCALAR__, __TECS_DIMENSIONS__), #N>
#define TECS_FORCE_FIELD_MEMBER_TPL(T, N) FieldRefT<TECS_FIELD_TYPE(T, __TECS_SCALAR__, __TECS_DIMENSIONS__)> N

/**
 * @brief Record field pairs for deferred type resolution.
 *
 * This macro stores the field type/name pairs in a `_tecs_fields` template
 * that is parameterized on (Scalar, dimensions).  The template is resolved
 * later by `RESOLVE_TECS_CONFIG()` - which must be called **last**, after
 * all overrides - to produce the final `Vector`, `Polarity`, `VectorView`,
 * `Fields`, and `ForceFields` using whatever `Scalar` and `dimensions`
 * are then in scope.
 *
 * @code
 *   struct MyConfig : DefaultSimulationConfig {
 *     CONFIG_SCALAR(double)
 *     CONFIG_DIMENSIONS(2)
 *     CONFIG_FIELDS(
 *       (Vector, position),
 *       (Polarity, polarity)
 *     )
 *     RESOLVE_TECS_CONFIG()
 *   };
 * @endcode
 */
#define CONFIG_FIELDS(...) \
  template<typename __TECS_SCALAR__, int __TECS_DIMENSIONS__> \
  struct __TECS_FIELDS_STRUCT__ { \
    using Vector = tecs::VectorN<__TECS_SCALAR__, __TECS_DIMENSIONS__>; \
    using Polarity = tecs::Polarity_<__TECS_SCALAR__>; \
    using VectorView = tecs::View<Vector>; \
    using Fields = tecs::detail::FieldList<FOR_EACH_PAIR(TECS_FIELD_TPL, __VA_ARGS__)>; \
    template<template<typename> typename FieldRefT = tecs::detail::GenericFieldRef> \
    struct ForceFields { \
      FIELDS_ITERATE(TECS_FORCE_FIELD_MEMBER_TPL, __VA_ARGS__) \
    }; \
  };

/**
 * @brief Resolve the deferred field config using the current Scalar/dimensions.
 *
 * Must be placed **last** in the config struct (after all overrides and
 * after `CONFIG_FIELDS`).  Defines `Vector`, `Polarity`, `VectorView`,
 * `Fields`, and `ForceFields` by instantiating `_tecs_fields` with the
 * final `Scalar` and `dimensions`.
 */
#define RESOLVE_TECS_CONFIG() \
  using __TECS_RESOLVED__ = __TECS_FIELDS_STRUCT__<Scalar, dimensions>; \
  using Vector = __TECS_RESOLVED__::Vector; \
  using Polarity = __TECS_RESOLVED__::Polarity; \
  using VectorView = __TECS_RESOLVED__::VectorView; \
  using Fields = __TECS_RESOLVED__::Fields; \
  template<template<typename> typename FieldRefT = tecs::detail::GenericFieldRef> \
  using ForceFields = __TECS_RESOLVED__::template ForceFields<FieldRefT>;

  struct DefaultSimulationConfig {
    CONFIG_DIMENSIONS(3)
    CONFIG_SCALAR(float)


    CONFIG_RANDOM_POOL(Kokkos::Random_XorShift64_Pool)
    CONFIG_PAIR_FINDER(tecs::pair_finders::NaiveAllPairs)
    CONFIG_COM_FIXER(tecs::com_fixers::NoComFixer)
    CONFIG_INTEGRATOR(tecs::integrators::Heun)

    CONFIG_WRITER(tecs::io::HDF5_Writer)

    CONFIG_FIELDS(
      (Vector, position)
    )

    RESOLVE_TECS_CONFIG()
  };

/**
 * @brief Create a simulation config struct in one go.
 *
 * @code
 *   CREATE_SIMULATION_CONFIG(MyConfig,
 *     CONFIG_PAIR_FINDER(pair_finders::NaiveDelaunay)
 *     CONFIG_SCALAR(double)
 *     CONFIG_DIMENSIONS(2)
 *     CONFIG_FIELDS((Vector, position), (Polarity, polarity))
 *   );
 * @endcode
 */
#define CREATE_SIMULATION_CONFIG(__NAME__, ...) \
  struct __NAME__ : public tecs::DefaultSimulationConfig { \
    __VA_ARGS__ \
    RESOLVE_TECS_CONFIG() \
  };

} // namespace tecs

#endif // TECS_SIMULATION_CONFIG
