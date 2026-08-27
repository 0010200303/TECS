#ifndef TECS_BENCHMARK_NAIVE_ALL_PAIRS_PARALLEL_REDUCE_HPP
#define TECS_BENCHMARK_NAIVE_ALL_PAIRS_PARALLEL_REDUCE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveAllPairsParallelReduce {
    using positions_view_type = PositionsView;

    struct Settings { };

    NaiveAllPairsParallelReduce(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;

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
        "naive_all_pairs_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_drag_i = Scalar(0);
          typename PositionsView::value_type total_velocity_i{0};

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

              const auto distance = Kokkos::sqrt(distance_squared);

              Scalar pairwise_drag = Scalar(0);

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
} // namespace tecs::benchmark

#endif // TECS_BENCHMARK_NAIVE_ALL_PAIRS_PARALLEL_REDUCE_HPP
