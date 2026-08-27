#ifndef TECS_FORCES_KERNEL_FUSER_HPP
#define TECS_FORCES_KERNEL_FUSER_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace tecs::detail {

  /**
   * @brief Combines multiple force functors with the same tag into a single
   *        callable, so they can be evaluated in one kernel launch.
   *
   * Instead of launching one parallel_for per force, KernelFuser chains them
   * via recursive inheritance: calling `operator()` runs the first force, then
   * delegates to the base for the rest. This reduces kernel launch overhead.
   *
   * @tparam Tag    The force tag (GenericForceTag, PairwiseForceTag, etc.).
   * @tparam Forces The individual force functors to fuse.
   */
  template<typename Tag, typename... Forces>
  struct KernelFuser;

  /// @brief Empty (base-case) KernelFuser - no-op.
  template<typename Tag>
  struct KernelFuser<Tag> {
    using tag = Tag;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&...) const { }
  };

  /// @brief Non-empty KernelFuser - calls the first force, then recurses.
  template<typename Tag, typename FirstForce, typename... RestForces>
  struct KernelFuser<Tag, FirstForce, RestForces...> : KernelFuser<Tag, RestForces...> {
    using base_type = KernelFuser<Tag, RestForces...>;

    FirstForce force;

    KernelFuser(FirstForce first, RestForces... rest)
      : base_type(std::move(rest)...), force(std::move(first)) { }

    /// @brief Run all fused forces in sequence for the given arguments.
    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&... args) const {
      force(static_cast<Args&&>(args)...);
      static_cast<const base_type&>(*this)(static_cast<Args&&>(args)...);
    }
  };

  /// @brief CTAD deduction guide.
  template<typename Tag, typename... Forces>
  KernelFuser(Tag, Forces...) -> KernelFuser<Tag, Forces...>;

  // --- SFINAE helpers ---

  template<typename T, typename = void>
  struct has_force_member : std::false_type { };

  template<typename T>
  struct has_force_member<T, std::void_t<decltype(std::declval<T>().force)>> : std::true_type { };

  template<typename T, typename = void>
  struct has_tag_member : std::false_type { };

  template<typename T>
  struct has_tag_member<T, std::void_t<typename T::tag>> : std::true_type { };

  template<typename T, typename = void>
  struct is_nullary_callable : std::false_type { };

  template<typename T>
  struct is_nullary_callable<T, std::void_t<decltype(std::declval<T>()())>> : std::true_type { };

  /**
   * @brief Unwrap a force expression.
   *
   * If the input is a nullary lambda that returns a TaggedForce, call it.
   * If it already has a `tag`, pass through directly.
   */
  template<typename T>
  decltype(auto) unwrap_force(T&& force) {
    if constexpr (has_tag_member<std::decay_t<T>>::value)
      return std::forward<T>(force);
    else if constexpr (is_nullary_callable<T>::value)
      return std::forward<T>(force)();
    else
      return std::forward<T>(force);
  }

  /**
   * @brief Extract forces matching a specific tag into a tuple.
   *
   * If the force has the requested tag, extract it (unwrapping the
   * TaggedForce wrapper). Otherwise return an empty tuple.
   */
  template<typename Tag, typename Force>
  auto collect_tagged_force(Force&& force) {
    if constexpr (std::is_same_v<typename std::decay_t<Force>::tag, Tag>) {
      if constexpr (has_force_member<std::decay_t<Force>>::value) {
        using pure_force_t = std::decay_t<decltype(std::forward<Force>(force).force)>;
        return std::tuple<pure_force_t>(std::forward<Force>(force).force);
      }
      else {
        return std::tuple<std::decay_t<Force>>(std::forward<Force>(force));
      }
    }
    else {
      return std::tuple<>{};
    }
  }

  /**
   * @brief Fuse all forces with a given tag into a single KernelFuser.
   *
   * Filters the input forces by tag, unwraps them, and builds a KernelFuser.
   */
  template<typename Tag, typename... Forces>
  auto fuse_forces_for_tag(Forces&&... forces) {
    return std::apply([](auto&&... kernels) {
      return KernelFuser<Tag, std::decay_t<decltype(kernels)>...> {
        std::forward<decltype(kernels)>(kernels)...
      };
    }, std::tuple_cat(collect_tagged_force<Tag>(unwrap_force(std::forward<Forces>(forces)))...));
  }

  /// @brief Helper: wrap a value in a tuple, or return empty tuple if empty.
  template<typename T>
  auto as_tuple_if_not_empty(T&& value) {
    if constexpr (std::is_empty_v<std::decay_t<T>>)
      return std::tuple<>{};
    else
      return std::tuple<std::decay_t<T>>(std::forward<T>(value));
  }

  /**
   * @brief Fuse all forces into a tuple of KernelFusers, grouped by tag.
   *
   * The result is a tuple of up to four KernelFusers (one per tag type),
   * each fusing all forces with that tag. Empty fusers are omitted.
   */
  template<typename... Forces>
  auto fuse_forces(Forces&&... forces) {
    return std::tuple_cat(
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::GenericForceTag>(std::forward<Forces>(forces)...)),
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::PairwiseForceTag>(std::forward<Forces>(forces)...)),
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::UpdateFuncTag>(std::forward<Forces>(forces)...)),
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::LinkForceTag>(std::forward<Forces>(forces)...))
    );
  }
} // namespace tecs::detail

#endif // TECS_FORCES_KERNEL_FUSER_HPP
