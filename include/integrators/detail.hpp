#ifndef TECS_INTEGRATORS_DETAIL_HPP
#define TECS_INTEGRATORS_DETAIL_HPP

#include <type_traits>

namespace tecs::detail {

  /**
   * @brief A type-safe, variadic tuple of views used throughout the integrator
   *        and force-evaluation pipeline.
   *
   * ViewPack stores one view per simulation field (positions, velocities,
   * forces, ...) and provides three calling conventions:
   *
   * - **`apply(f)`**               - calls `f` with all views as arguments (device + host).
   * - **`apply_host(f)`**          - host-only apply (for resize, realloc, etc.).
   * - **`zip_apply(f, other...)`** - calls `f` element-wise across multiple
   *   packs of the same size (used in integrator steps to update positions
   *   from force increments).
   *
   * @tparam Views The view types that hold the simulation fields.
   */
  template<typename... Views>
  struct ViewPack;

  /// @brief Compile-time size of a ViewPack.
  template<typename Pack>
  struct view_pack_size;

  template<typename... Views>
  struct view_pack_size<ViewPack<Views...>> {
    static constexpr std::size_t value = sizeof...(Views);
  };

  /// @brief Helper variable template for `view_pack_size`.
  template<typename Pack>
  inline constexpr std::size_t view_pack_size_v =
    view_pack_size<std::remove_cv_t<std::remove_reference_t<Pack>>>::value;

  /**
   * @brief Empty ViewPack - base case of the recursive inheritance.
   *
   * `apply` simply calls the functor with no arguments.
   * `zip_apply` is a no-op (no views to iterate over).
   */
  template<>
  struct ViewPack<> {
    static constexpr std::size_t size = 0;

    ViewPack() = default;

    /// @brief Invoke `f` with no arguments.
    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) const {
      return static_cast<F&&>(f)();
    }

    /// @copydoc apply
    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) {
      return static_cast<F&&>(f)();
    }

    /// @brief Host-only apply (no arguments).
    template<typename F>
    decltype(auto) apply_host(F&& f) const {
      return static_cast<F&&>(f)();
    }

    /// @copydoc apply_host
    template<typename F>
    decltype(auto) apply_host(F&& f) {
      return static_cast<F&&>(f)();
    }

    /// @brief Zip over nothing - no-op.
    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&&, OtherPacks&&...) { }

    /// @copydoc zip_apply
    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&&, OtherPacks&&...) const { }
  };

  /**
   * @brief Non-empty ViewPack - stores one view and inherits from the rest.
   *
   * Uses recursive inheritance so each level stores exactly one view and
   * delegates to its base for the remaining views. This enables:
   * - `first()`                      - access the first view.
   * - `get<I>()`                     - compile-time index access.
   * - `apply(f)`                     - calls `f(view_0, view_1, …, view_n)`.
   * - `zip_apply(f, other_packs...)` - element-wise application across packs.
   */
  template<typename FirstView, typename... RestViews>
  struct ViewPack<FirstView, RestViews...> : ViewPack<RestViews...> {
    using base_type = ViewPack<RestViews...>;
    static constexpr std::size_t size = 1 + base_type::size;

    /// The first view (head) in this pack.
    FirstView first_value;

    ViewPack() = default;

    /// @brief Construct from individual views.
    KOKKOS_INLINE_FUNCTION
    ViewPack(const FirstView& first, const RestViews&... rest)
      : base_type(rest...)
      , first_value(first) { }

    /// @brief Construct from a first view and an existing base pack.
    KOKKOS_INLINE_FUNCTION
    ViewPack(const FirstView& first, const base_type& rest)
      : base_type(rest)
      , first_value(first) { }

    /// @brief Access the first view.
    KOKKOS_INLINE_FUNCTION
    FirstView& first() {
      return first_value;
    }

    /// @copydoc first()
    KOKKOS_INLINE_FUNCTION
    const FirstView& first() const {
      return first_value;
    }

    /// @brief Access the view at compile-time index `I`.
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) get() {
      if constexpr (I == 0)
        return first_value;
      else
        return static_cast<base_type&>(*this).template get<I - 1>();
    }

    /// @copydoc get
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) get() const {
      if constexpr (I == 0)
        return first_value;
      else
        return static_cast<const base_type&>(*this).template get<I - 1>();
    }

    /// @brief Invoke `f(first_value, rest_values...)`.
    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) {
      return static_cast<base_type&>(*this).apply(
        [&](auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    /// @copydoc apply
    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) const {
      return static_cast<const base_type&>(*this).apply(
        [&](const auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    /// @brief Host-only apply with all views.
    template<typename F>
    decltype(auto) apply_host(F&& f) const {
      return static_cast<const base_type&>(*this).apply_host(
        [&](const auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    /// @copydoc apply_host
    template<typename F>
    decltype(auto) apply_host(F&& f) {
      return static_cast<base_type&>(*this).apply_host(
        [&](auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    /**
     * @brief Apply a functor element-wise across this and other packs.
     *
     * Calls `f(this[0], other_1[0], other_2[0], ...)` for the first element,
     * then recurses for the remaining elements. All packs must have the
     * same size (enforced by static_assert).
     */
    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&& f, OtherPacks&&... others) {
      static_assert(
        ((view_pack_size_v<OtherPacks> == size) && ...),
        "ViewPack::zip_apply pack size mismatch"
      );

      f(first_value, others.first()...);
      base_type::zip_apply(
        std::forward<F>(f),
        static_cast<base_type&>(others)...
      );
    }

    /// @copydoc zip_apply
    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&& f, OtherPacks&&... others) const {
      static_assert(
        ((view_pack_size_v<OtherPacks> == size) && ...),
        "ViewPack::zip_apply pack size mismatch"
      );

      f(first_value, others.first()...);
      static_cast<const base_type&>(*this).zip_apply(
        std::forward<F>(f),
        static_cast<const base_type&>(others)...
      );
    }
  };

  /// @brief Free-function accessor for the first view in a ViewPack.
  template<typename FirstView, typename... RestViews>
  KOKKOS_INLINE_FUNCTION
  FirstView& first(ViewPack<FirstView, RestViews...>& pack) {
    return pack.first();
  }

  /// @copydoc first
  template<typename FirstView, typename... RestViews>
  KOKKOS_INLINE_FUNCTION
  const FirstView& first(const ViewPack<FirstView, RestViews...>& pack) {
    return pack.first();
  }

  /// @brief Extract the first type from a parameter pack.
  template<typename First, typename...>
  struct first_type {
    using type = First;
  };

  /// @brief Helper alias for `first_type`.
  template<typename... Views>
  using first_type_t = typename first_type<Views...>::type;

  /**
   * @brief An array of N ViewPacks, used by multi-stage integrators (Euler,
   *        Heun).
   *
   * Stage 0 holds the current simulation state. Stages 1 ... N - 1 hold
   * intermediate force deltas or predicted states. On construction,
   * stages 1 ... N - 1 are automatically created as "mirror" ViewPacks (same
   * labels and sizes, freshly allocated).
   *
   * @tparam N     Number of stages (e.g. 2 for Euler, 4 for Heun).
   * @tparam Views The view types that hold the simulation fields.
   */
  template<int N, typename... Views>
  struct StagePack {
    static_assert(N > 0, "StagePack must contain at least one stage");

    /// The N stage ViewPacks.
    ViewPack<Views...> stages[N];

    template<typename FirstView, typename... RestViews>
    static auto make_mirror_view_pack_impl(const ViewPack<FirstView, RestViews...>& pack) {
      using mirror_first_type = FirstView;

      if constexpr (sizeof...(RestViews) == 0) {
        return ViewPack<mirror_first_type>(
          FirstView(pack.first().label(), pack.first().extent(0))
        );
      } else {
        return ViewPack<mirror_first_type, RestViews...>(
          FirstView(pack.first().label(), pack.first().extent(0)),
          make_mirror_view_pack_impl(static_cast<const ViewPack<RestViews...>&>(pack))
        );
      }
    }

    /// @brief Create a new ViewPack with the same structure (labels & sizes).
    static auto make_mirror_view_pack(const ViewPack<Views...>& pack) {
      return make_mirror_view_pack_impl(pack);
    }

    /**
     * @brief Construct all N stages.
     *
     * Stage 0 uses the provided pack directly; stages 1 ... N - 1 are mirrors
     * (same labels and sizes but freshly allocated).
     */
    StagePack(const ViewPack<Views...>& stage) {
      stages[0] = stage;
      for (int i = 1; i < N; ++i) {
        stages[i] = make_mirror_view_pack(stage);
      }
    }

    /// @brief Access the i-th stage ViewPack.
    KOKKOS_INLINE_FUNCTION
    ViewPack<Views...>& operator[](int i) {
      return stages[i];
    }

    /// @copydoc operator[]
    KOKKOS_INLINE_FUNCTION
    const ViewPack<Views...>& operator[](int i) const {
      return stages[i];
    }
  };
} // namespace tecs::detail

#endif // TECS_INTEGRATORS_DETAIL_HPP
