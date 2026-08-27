#ifndef TECS_BENCHMARK_NAIVE_GABRIEL_FOR_HPP
#define TECS_BENCHMARK_NAIVE_GABRIEL_FOR_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabrielFor {
    using positions_view_type = PositionsView;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    NaiveGabrielFor(
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
          Scalar total_drag_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          for (int j = 0; j < agent_count; ++j) {
            if (i == j)
              continue;

            const auto position_j = input_positions(j);
            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();

            if (distance_squared >= cutoff_distance_squared)
              continue;

            const auto midpoint = position_i - displacement * Scalar(0.5);
            const auto radius_squared = distance_squared * Scalar(0.25) * gabriel_coefficient_squared;
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
              continue;

            const auto distance = Kokkos::sqrt(distance_squared);

            Scalar pairwise_drag = 0.0;

            auto generator = random_pool.get_state();
            in_view_pack.apply([&](auto&... views) {
              total_delta_i.apply([&](auto&... deltas) {
                force(
                  is_full_step, i, j, displacement, distance, generator, pairwise_drag,
                  ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
                );
              });
            });
            random_pool.free_state(generator);

            total_drag_i += pairwise_drag;
            total_velocity_i += pairwise_drag * old_velocities(j);
          }

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

#endif // TECS_BENCHMARK_NAIVE_GABRIEL_FOR_HPP
