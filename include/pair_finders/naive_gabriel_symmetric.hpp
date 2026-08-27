#ifndef TECS_PAIR_FINDERS_NAIVE_GABRIEL_SYMMETRIC_HPP
#define TECS_PAIR_FINDERS_NAIVE_GABRIEL_SYMMETRIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace tecs::pair_finders {

  /**
   * @brief Naive O(N³) Gabriel-graph pair finder with symmetric evaluation.
   *
   * Evaluates each unordered pair @f$(i,j)@f$ with @f$j < i@f$ exactly once,
   * and for each candidate pair tests whether the Gabriel condition is
   * satisfied:  the open ball with diameter @f$\overline{ij}@f$ (scaled by
   * the Gabriel coefficient) must contain no other particle @f$k@f$.
   *
   * The pair is processed only if it lies within the cutoff distance *and*
   * passes the Gabriel test.  Force deltas are computed once and atomically
   * applied to both @f$i@f$ (add) and @f$j@f$ (subtract).  Drag and
   * velocity terms are likewise accumulated atomically for both particles
   * and normalised in a finalisation pass.
   *
   * Because the Gabriel test itself is O(N), the overall complexity is
   * O(N³).  This pair finder is intended for small systems or for baseline
   * correctness verification against the binned variant.
   *
   * @tparam PositionsView Kokkos view type for particle positions.
   * @tparam Scalar         Floating-point type (e.g., \c double).
   * @tparam dimensions     Number of spatial dimensions (1, 2, or 3).
   */
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabrielSymmetric {
    using positions_view_type = PositionsView;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    /**
     * @brief Constructs the naive symmetric Gabriel pair finder.
     * @param agent_count_    Number of particles.
     * @param cutoff_distance Interaction cutoff distance.
     * @param settings        Configuration (Gabriel coefficient).
     */
    NaiveGabrielSymmetric(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings
    ) : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient_squared(settings.gabriel_coefficient * settings.gabriel_coefficient) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient_squared;

    /** @brief Updates the number of particles. */
    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
    }

    /**
     * @brief Evaluates forces for all Gabriel-graph symmetric pairs using
     *        a naive O(N³) approach.
     *
     * For each particle @f$i@f$, a team-parallel inner loop iterates over
     * @f$j < i@f$.  For each candidate pair within the cutoff, the Gabriel
     * empty-ball test is performed by scanning all other particles @f$k@f$.
     * If the test passes, the force functor is invoked once and deltas,
     * drag, and velocity contributions are atomically accumulated for both
     * @f$i@f$ and @f$j@f$.  A separate finalisation kernel normalises the
     * drag correction.
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

      Kokkos::View<Scalar*> total_drag("ng_sym_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("ng_sym_total_velocity", agent_count);

      Kokkos::parallel_for(
        "naive_gabriel_symmetric_apply_force",
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

              // Gabriel empty-ball test: check if any k is inside the
              // scaled diametral ball of (i,j).
              const auto midpoint = position_i - displacement * Scalar(0.5);
              const auto radius_squared =
                distance_squared * Scalar(0.25) * gabriel_coefficient_squared;
              bool is_gabriel = true;
              for (int k = 0; k < agent_count; ++k) {
                if (k == i || k == j)
                  continue;
                const auto offset = input_positions(k) - midpoint;
                if (offset.length_squared() < radius_squared) {
                  is_gabriel = false;
                  break;
                }
              }
              if (!is_gabriel)
                return;

              const auto distance = Kokkos::sqrt(distance_squared);

              Scalar pairwise_drag = 0.0;

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
        "naive_gabriel_symmetric_apply_drag",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_NAIVE_GABRIEL_SYMMETRIC_HPP
