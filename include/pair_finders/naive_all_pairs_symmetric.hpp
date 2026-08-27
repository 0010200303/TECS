#ifndef TECS_PAIR_FINDERS_NAIVE_ALL_PAIRS_SYMMETRIC_HPP
#define TECS_PAIR_FINDERS_NAIVE_ALL_PAIRS_SYMMETRIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace tecs::pair_finders {

  /**
   * @brief Naive O(N²) all-pairs pair finder with symmetric force evaluation.
   *
   * Evaluates each unordered pair @f$(i,j)@f$ with @f$j < i@f$ exactly once.
   * For each particle @f$i@f$, a team-level parallel loop iterates over
   * @f$j \in [0, i)@f$.  The force contribution is computed once and the
   * resulting deltas are applied to both @f$i@f$ and @f$-j@f$ via atomic
   * operations.  Drag and velocity terms are likewise atomically accumulated
   * for both particles.
   *
   * This halves the number of pairwise evaluations compared to the
   * non-symmetric NaiveAllPairs, at the cost of atomics which may limit
   * throughput on certain devices.  It is well suited for simulations where
   * forces are anti-symmetric (action = -reaction) and the cutoff is large
   * relative to the domain, making binned approaches less beneficial.
   *
   * @tparam PositionsView Kokkos view type for particle positions.
   * @tparam Scalar         Floating-point type (e.g., \c double).
   * @tparam dimensions     Number of spatial dimensions (1, 2, or 3).
   */
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveAllPairsSymmetric {
    using positions_view_type = PositionsView;

    struct Settings { };

    /**
     * @brief Constructs the symmetric all-pairs pair finder.
     * @param agent_count_    Number of particles.
     * @param cutoff_distance Interaction cutoff distance.
     * @param settings        Unused settings placeholder (no configuration).
     */
    NaiveAllPairsSymmetric(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;

    /** @brief Updates the number of particles. */
    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
    }

    /**
     * @brief Evaluates forces for all symmetric pairs using a naive O(N²)
     *        approach.
     *
     * For each particle @f$i@f$, a team-parallel inner loop iterates over
     * @f$j < i@f$.  If @f$|\mathbf{r}_i - \mathbf{r}_j|@f$ is within the
     * cutoff, the user-supplied force functor is invoked once.  Deltas are
     * atomically added to @f$i@f$ and subtracted from @f$j@f$.  Drag and
     * velocity corrections are accumulated atomically and normalized in a
     * separate finalization kernel.
     *
     * @tparam RandomPool  Kokkos random pool type.
     * @tparam Force       Force functor type.
     * @tparam ForceFields Type list of field-view pairs.
     * @tparam Views       Additional view types.
     * @param in_view_pack   Pack of read-only views (first is positions).
     * @param out_view_pack  Pack of writable views (accumulated deltas + drag).
     * @param old_velocities Previous-step velocities for drag correction.
     * @param random_pool    Kokkos random pool for stochastic forces.
     * @param force          Force functor instance.
     * @param is_full_step   Whether this is a full (not half) step.
     */
    template<typename RandomPool, typename Force, typename ForceFields, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force,
      bool is_full_step
    ) {
      const auto& input_positions = in_view_pack.first();

      Kokkos::View<Scalar*> total_drag("nap_sym_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("nap_sym_total_velocity", agent_count);

      Kokkos::parallel_for(
        "naive_all_pairs_symmetric_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto position_i = input_positions(i);

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, i),
            [&](const int j) {
              const auto position_j = input_positions(j);
              const auto displacement = position_i - position_j;
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

              const auto distance = Kokkos::sqrt(distance_squared);

              Scalar pairwise_drag = Scalar(0);

              auto local_delta = detail::make_accumulator_pack(out_view_pack);
              auto generator = random_pool.get_state();
              in_view_pack.apply([&](auto&... views) {
                local_delta.apply([&](auto&... deltas) {
                  force(
                    is_full_step, i, j, displacement, distance, generator, pairwise_drag,
                    ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
                  );
                });
              });
              random_pool.free_state(generator);

              out_view_pack.apply([&](auto&... views) {
                local_delta.apply([&](auto&... values) {
                  ((Kokkos::atomic_add(&views(i), values)), ...);
                  ((Kokkos::atomic_add(&views(j), -values)), ...);
                });
              });
              Kokkos::atomic_add(&total_drag(i), pairwise_drag);
              Kokkos::atomic_add(&total_velocity(i),
                                 pairwise_drag * old_velocities(j));
              Kokkos::atomic_add(&total_drag(j), pairwise_drag);
              Kokkos::atomic_add(&total_velocity(j),
                                 pairwise_drag * old_velocities(i));
            }
          );
        }
      );

      Kokkos::parallel_for(
        "naive_all_pairs_symmetric_apply_drag",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_NAIVE_ALL_PAIRS_SYMMETRIC_HPP
