#ifndef TECS_PAIR_FINDERS_NAIVE_GABRIEL_HPP
#define TECS_PAIR_FINDERS_NAIVE_GABRIEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace tecs::pair_finders {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabriel {
    using positions_view_type = PositionsView;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    NaiveGabriel(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings
    ) : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient_squared(settings.gabriel_coefficient * settings.gabriel_coefficient) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient_squared;

    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
    }

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

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_drag_i = Scalar(0);
          typename PositionsView::value_type total_velocity_i{Scalar(0)};

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, agent_count),
            [&](const int j, auto& local_delta, auto& local_drag, auto& local_velocity) {
              if (i == j)
                return;

              const auto position_j = input_positions(j);
              const auto displacement = position_i - position_j;
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

              const auto midpoint = position_i - displacement * Scalar(0.5);
              const auto radius_squared = distance_squared * Scalar(0.25) * gabriel_coefficient_squared;
              for (int k = 0; k < agent_count; ++k) {
                if (k == i || k == j)
                  continue;

                const auto offset = input_positions(k) - midpoint;
                if (offset.length_squared() < radius_squared)
                  return;
              }

              const auto distance = Kokkos::sqrt(distance_squared);

              Scalar pairwise_drag = 0.0;

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

              local_drag += pairwise_drag;
              local_velocity += pairwise_drag * old_velocities(j);
            },
            Kokkos::Sum<decltype(total_delta_i)>(total_delta_i),
            Kokkos::Sum<Scalar>(total_drag_i),
            Kokkos::Sum<typename PositionsView::value_type>(total_velocity_i)
          );

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              out_view_pack.apply([&](auto&... views) {
                total_delta_i.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });

              out_view_pack.first()(i) += total_velocity_i /
                Kokkos::fmax(total_drag_i, Kokkos::Experimental::epsilon_v<Scalar>);
            }
          );
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_NAIVE_GABRIEL_HPP
