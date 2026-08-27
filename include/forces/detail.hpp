#ifndef TECS_FORCES_DETAIL_HPP
#define TECS_FORCES_DETAIL_HPP

#include <type_traits>
#include <utility>

namespace tecs::detail {

  // --- Force-type tags used for tag dispatch ---

  /// @brief Tag for per-agent (generic) force functors.
  struct GenericForceTag { };
  /// @brief Tag for pairwise interaction force functors.
  struct PairwiseForceTag { };
  /// @brief Tag for update functions (side effects, no force output).
  struct UpdateFuncTag { };
  /// @brief Tag for link force functors.
  struct LinkForceTag { };

  /**
   * @brief Wraps a force lambda together with its tag type.
   *
   * Created by the `operator|` mechanism used in the force definition macros
   * (e.g. `GENERIC_FORCE_IMPL`, `PAIRWISE_FORCE_IMPL`).
   */
  template<typename Tag, typename Force>
  struct TaggedForce {
    Force force;

    KOKKOS_INLINE_FUNCTION
    TaggedForce(Force force_) : force(force_) { }

    using tag = Tag;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&... args) const {
      force(static_cast<Args&&>(args)...);
    }
  };

  /**
   * @brief Tagger that enables the `force | tag` and `tag | force` syntax.
   */
  template<typename Tag>
  struct ForceTagger {
    template<typename Force>
    friend auto operator|(ForceTagger, Force force) {
      return TaggedForce<Tag, Force>(force);
    }

    template<typename Force>
    friend auto operator|(Force force, ForceTagger) {
      return TaggedForce<Tag, Force>(force);
    }
  };

  /// @name Force tag instances (used by the force macros).
  ///@{
  constexpr ForceTagger<GenericForceTag> generic_force{};
  constexpr ForceTagger<PairwiseForceTag> pairwise_force{};
  constexpr ForceTagger<UpdateFuncTag> update_func{};
  constexpr ForceTagger<LinkForceTag> link_force{};
  ///@}



  /**
   * @brief A single slot in an AccumulatorPack, identified by a compile-time index.
   *
   * Used inside link-force evaluation to accumulate force contributions before
   * applying them atomically to the output views.
   */
  template<std::size_t Index, typename T>
  struct AccumulatorSlot {
    T value{};

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot() = default;

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot(const T& v) : value(v) { }

    KOKKOS_INLINE_FUNCTION
    T& get() {
      return value;
    }

    KOKKOS_INLINE_FUNCTION
    const T& get() const {
      return value;
    }

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot& operator+=(const AccumulatorSlot& other) {
      value += other.value;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot& operator+=(const T& other) {
      value += other;
      return *this;
    }
  };

  /**
   * @brief A pack of AccumulatorSlots, one per simulation field view.
   *
   * Inherits from each slot so slots are laid out optimally.
   * Supports per-field access via `get<I>()`, fold `+=`, and `apply(f)`.
   */
  template<typename... Slots>
  struct AccumulatorPack : Slots... {
    public:
      KOKKOS_INLINE_FUNCTION
      AccumulatorPack() = default;

      KOKKOS_INLINE_FUNCTION
      AccumulatorPack(Slots... types) : Slots(types)... { }

      KOKKOS_INLINE_FUNCTION
      AccumulatorPack& operator+=(const AccumulatorPack& other) {
        ((static_cast<Slots&>(*this) += static_cast<const Slots&>(other)), ...);
        return *this;
      }

      template<typename Func>
      KOKKOS_INLINE_FUNCTION
      decltype(auto) apply(Func&& func) {
        return func(static_cast<Slots&>(*this).get()...);
      }

      template<typename Func>
      KOKKOS_INLINE_FUNCTION
      decltype(auto) apply(Func&& func) const {
        return func(static_cast<const Slots&>(*this).get()...);
      }

      template<std::size_t I>
      KOKKOS_INLINE_FUNCTION
      decltype(auto) get() {
        return get_slot_impl<I>(*this).get();
      }

      template<std::size_t I>
      KOKKOS_INLINE_FUNCTION
      decltype(auto) get() const {
        return get_slot_impl<I>(*this).get();
      }

    private:
      template<std::size_t I, typename T>
      KOKKOS_INLINE_FUNCTION
      static AccumulatorSlot<I, T>& get_slot_impl(AccumulatorSlot<I, T>& slot) {
        return slot;
      }

      template<std::size_t I, typename T>
      KOKKOS_INLINE_FUNCTION
      static const AccumulatorSlot<I, T>& get_slot_impl(const AccumulatorSlot<I, T>& slot) {
        return slot;
      }
  };

  /// @brief Helper to create an AccumulatorPack matching the types in a ViewPack.
  template<typename... Views, std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  static auto make_accumulator_pack_impl(std::index_sequence<Is...>) {
    return AccumulatorPack<AccumulatorSlot<Is, typename Views::value_type>...>(
      AccumulatorSlot<Is, typename Views::value_type>{}...
    );
  }

  /// @copydoc make_accumulator_pack_impl
  template<typename... Views>
  KOKKOS_INLINE_FUNCTION
  static auto make_accumulator_pack(const ViewPack<Views...>& pack) {
    return make_accumulator_pack_impl<Views...>(std::index_sequence_for<Views...>{});
  }



  /**
   * @brief Field references passed to a generic (per-agent) force functor.
   *
   * Provides `self` (the current value) and `delta` (write delta).
   */
  template<typename T>
  struct GenericFieldRef {
    const T& self;   ///< Current value of the field for this agent.
    T& delta;        ///< Delta that the force can write to.
  };

  /**
   * @brief Field references passed to a pairwise force functor.
   *
   * Provides `self` and `other` (the two interacting agents), plus `delta`
   * for writing the force delta.
   */
  template<typename T>
  struct PairwiseFieldRef {
    const T& self;   ///< Current value for agent i.
    const T& other;  ///< Current value for agent j.
    T& delta;        ///< Delta to write.
  };

  /**
   * @brief Field references passed to a link force functor.
   *
   * Provides both endpoint values (`a` and `b`) and separate delta slots
   * (`delta_a`, `delta_b`) so each endpoint can receive its own contribution.
   */
  template<typename T>
  struct LinkFieldRef {
    const T& a;      ///< Current value for the first endpoint.
    const T& b;      ///< Current value for the second endpoint.
    T& delta_a;      ///< Delta for endpoint a.
    T& delta_b;      ///< Delta for endpoint b.
  };
} // namespace tecs::detail

namespace Kokkos {
  /// @brief Reduction identity for AccumulatorSlot (zero-initialised).
  template<typename T>
  struct reduction_identity<tecs::detail::AccumulatorSlot<0, T>> {
    KOKKOS_INLINE_FUNCTION
    static constexpr tecs::detail::AccumulatorSlot<0, T> sum() {
      return tecs::detail::AccumulatorSlot<0, T>{};
    }
  };

  /// @brief Reduction identity for AccumulatorPack (all slots zero-initialised).
  template<typename... Slots>
  struct reduction_identity<tecs::detail::AccumulatorPack<Slots...>> {
    KOKKOS_INLINE_FUNCTION
    static constexpr tecs::detail::AccumulatorPack<Slots...> sum() {
      return tecs::detail::AccumulatorPack<Slots...>{};
    }
  };
} // namespace Kokkos

#endif // TECS_FORCES_DETAIL_HPP
