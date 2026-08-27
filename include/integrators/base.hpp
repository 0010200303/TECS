#ifndef TECS_INTEGRATORS_BASE_HPP
#define TECS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "detail.hpp"
#include "../forces/detail.hpp"

namespace tecs::integrators {

  /**
   * @brief Base class for numerical integrators in the simulation framework.
   *
   * Manages the simulation state across multiple integration stages (used by
   * Runge–Kutta-like schemes), evaluates force fields, and can handle centre-of-mass
   * fixing. Concrete integrators like Euler or Heun inherit from this and
   * implement the actual time-stepping logic.
   *
   * @tparam PairFinder          Algorithm for finding pairwise interactions.
   * @tparam ComFixer            Centre-of-mass fixing strategy.
   * @tparam N                   Number of integration stages.
   * @tparam GenericForceFields  Pack of field references for per-agent forces.
   * @tparam PairwiseForceFields Pack of field references for pair forces.
   * @tparam LinkForceFields     Pack of field references for link forces.
   * @tparam Views               View types that hold the simulation data.
   */
  template<
    typename PairFinder,
    typename ComFixer,
    const unsigned int N,
    typename GenericForceFields,
    typename PairwiseForceFields,
    typename LinkForceFields,
    typename... Views
  >
  struct Base {
    /**
     * @brief Construct the integrator base.
     *
     * @param agent_count_  Number of active agents.
     * @param capacity      Allocated capacity for agent data.
     * @param pair_finder_  Reference to the pair-finding strategy.
     * @param com_fixer_    Reference to the centre-of-mass fixer.
     * @param links_        Device view of links between agents.
     * @param views         The simulation field views for stage 0.
     */
    Base(
      unsigned int agent_count_,
      unsigned int capacity,
      PairFinder& pair_finder_,
      ComFixer& com_fixer_,
      Kokkos::View<Link*> links_,
      Views... views
    ) : agent_count(agent_count_)
      , pair_finder(pair_finder_)
      , com_fixer(com_fixer_)
      , stage_pack(detail::ViewPack<Views...>(views...))
      , old_velocities("integrator_base_old_velocities", capacity)
      , links(links_) { }
    
    using PositionsView = typename PairFinder::positions_view_type;

    /// @brief The pair-finding algorithm (e.g. all pairs, delaunay, gabriel).
    PairFinder& pair_finder;
    /// @brief Centre-of-mass fixing algorithm.
    ComFixer& com_fixer;

    /// Number of active agents.
    unsigned int agent_count;
    /// The N-stage view pack: stage 0 = current simulation state, stages 1..N-1 = auxiliaries.
    detail::StagePack<N, Views...> stage_pack;
    /// Velocities from the last completed step (used by com fixers if enabled).
    PositionsView old_velocities;

    /// Links between agents.
    Kokkos::View<Link*> links;

    /**
     * @brief Replace the stage-0 views (current state) with new ones.
     *
     * This is needed when the simulation reallocates its data views.
     */
    template<typename... NewViews>
    void reattach_stage_0(const NewViews&... new_views) {
      stage_pack[0] = detail::ViewPack<Views...>(new_views...);
    }

    /**
     * @brief Resize all auxiliary stage views and the old-velocities buffer.
     * @param value New capacity.
     */
    inline void set_capacity(const unsigned int value) {
      Kokkos::resize(old_velocities, value);

      // resize all views from auxiliary stages
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((stage_pack[Is + 1].apply_host([&](auto&... views) {
          ((Kokkos::realloc(views, value)), ...);
        })), ...);
      }(std::make_index_sequence<N - 1>{});
    }

    /// @brief Update the active agent count.
    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
    }

    /**
     * @brief Evaluate a per-agent (generic) force in parallel.
     *
     * Calls the force functor once per agent, providing the current value
     * (via `in_view_pack`) and allowing it to write deltas (via `out_view_pack`).
     * Each agent gets its own random generator state.
     */
    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::GenericForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_generic_force",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          auto generator = random_pool.get_state();
          in_view_pack.apply([&](auto&... in_views) {
            out_view_pack.apply([&](auto&... out_views) {
              force(
                is_full_step,
                i,
                generator,
                GenericForceFields{detail::GenericFieldRef{in_views(i), out_views(i)}...}
              );
            });
          });
          random_pool.free_state(generator);
        }
      );
    }

    /**
     * @brief Evaluate a pairwise force using the pair finder.
     *
     * Delegates to `pair_finder.evaluate_force()`, which iterates over
     * neighbour pairs and computes interactions.
     */
    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::PairwiseForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      pair_finder.template evaluate_force<RandomPool, Force, PairwiseForceFields>(
        in_view_pack, out_view_pack, old_velocities, random_pool, force, is_full_step
      );
    }

    /**
     * @brief Evaluate an update function (no force output, arbitrary side effects).
     *
     * Useful for things like birth/death, lineage tracking, or polarity updates.
     */
    template<typename RandomPool, typename Func>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Func function,
      detail::UpdateFuncTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_update_func",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          auto generator = random_pool.get_state();
          function(i, generator);
          random_pool.free_state(generator);
        }
      );
    }

    /**
     * @brief Evaluate a force that acts along links between agents.
     *
     * Each link force receives the two endpoint values and can write
     * deltas into per-agent accumulators. The deltas are applied
     * atomically to avoid race conditions.
     */
    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::LinkForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_link_force",
        links.extent(0),
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          Link link = links(i);
          if (link.a == link.b)
            return;

          auto generator = random_pool.get_state();
          [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            auto accumulator_a = detail::make_accumulator_pack<Views...>(in_view_pack);
            auto accumulator_b = detail::make_accumulator_pack<Views...>(in_view_pack);

            force(is_full_step, link, generator, LinkForceFields{
              detail::LinkFieldRef{
                in_view_pack.template get<Is>()(link.a),
                in_view_pack.template get<Is>()(link.b),
                accumulator_a.template get<Is>(),
                accumulator_b.template get<Is>()
              }...
            });

            (Kokkos::atomic_add(&out_view_pack.template get<Is>()(link.a), accumulator_a.template get<Is>()), ...);
            (Kokkos::atomic_add(&out_view_pack.template get<Is>()(link.b), accumulator_b.template get<Is>()), ...);
          }(std::index_sequence_for<Views...>{});
          random_pool.free_state(generator);
        }
      );
    }

    /**
     * @brief Route one force to the correct implementation based on its tag.
     */
    template<typename RandomPool, typename Force>
    void evaluate_force_one(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      evaluate_force_impl(is_full_step, random_pool, force, typename Force::tag{}, in_view_pack, out_view_pack);
    }

    /**
     * @brief Evaluate all provided forces in sequence (fold expression).
     *
     * Each force is dispatched to the correct implementation based on its
     * `tag` type (GenericForceTag, PairwiseForceTag, UpdateFuncTag, LinkForceTag).
     */
    template<typename RandomPool, typename... Forces>
    void evaluate_forces(
      bool is_full_step,
      RandomPool& random_pool,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      Forces... forces
    ) {
      (evaluate_force_one(is_full_step, random_pool, forces, in_view_pack, out_view_pack), ...);
    }
  };
} // namespace tecs::integrators

#endif // TECS_INTEGRATORS_BASE_HPP
