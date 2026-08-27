#ifndef TECS_BENCHMARK_NAIVE_GABRIEL_PARALLEL_REDUCE_DOUBLE_HPP
#define TECS_BENCHMARK_NAIVE_GABRIEL_PARALLEL_REDUCE_DOUBLE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabrielParallelReduceDouble {
    using positions_view_type = PositionsView;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    NaiveGabrielParallelReduceDouble(
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

      Kokkos::View<Scalar*> total_drag("gabriel_pr_double_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("gabriel_pr_double_total_velocity", agent_count);

      Kokkos::parallel_for(
        "naive_gabriel_pr_double_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto position_i = input_positions(i);

          auto delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar drag_i = Scalar(0);
          typename PositionsView::value_type vel_i{0};

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, i),
            [&](const int j, auto& local_delta_i, auto& local_drag_i, auto& local_vel_i) {
              const auto position_j = input_positions(j);
              const auto displacement = position_i - position_j;
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

              const auto midpoint = position_i - displacement * Scalar(0.5);
              const auto radius_squared =
                distance_squared * Scalar(0.25) * gabriel_coefficient_squared;
              bool is_gabriel = true;
              Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team_member, agent_count),
                [&](const int k, bool& valid) {
                  if (k == i || k == j)
                    return;
                  const auto offset = input_positions(k) - midpoint;
                  if (offset.length_squared() < radius_squared)
                    valid = false;
                },
                Kokkos::LAnd<bool>(is_gabriel)
              );

              if (!is_gabriel)
                return;

              const auto distance = Kokkos::sqrt(distance_squared);

              {
                Scalar pairwise_drag = Scalar(0);
                auto tmp_delta = detail::make_accumulator_pack(out_view_pack);
                auto generator = random_pool.get_state();
                in_view_pack.apply([&](auto&... views) {
                  tmp_delta.apply([&](auto&... deltas) {
                    force(
                      is_full_step, i, j, displacement, distance, generator, pairwise_drag,
                      ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
                    );
                  });
                });
                random_pool.free_state(generator);

                local_delta_i += tmp_delta;
                local_drag_i += pairwise_drag;
                local_vel_i += pairwise_drag * old_velocities(j);
              }

              {
                const auto displacement_ji = -displacement;
                const auto distance_ji = distance;

                Scalar pairwise_drag_ji = Scalar(0);
                auto tmp_delta = detail::make_accumulator_pack(out_view_pack);
                auto generator = random_pool.get_state();
                in_view_pack.apply([&](auto&... views) {
                  tmp_delta.apply([&](auto&... deltas) {
                    force(
                      is_full_step, j, i, displacement_ji, distance_ji, generator, pairwise_drag_ji,
                      ForceFields{detail::PairwiseFieldRef{views(j), views(i), deltas}...}
                    );
                  });
                });
                random_pool.free_state(generator);

                out_view_pack.apply([&](auto&... views) {
                  tmp_delta.apply([&](auto&... values) {
                    ((Kokkos::atomic_add(&views(j), values)), ...);
                  });
                });
                Kokkos::atomic_add(&total_drag(j), pairwise_drag_ji);
                Kokkos::atomic_add(&total_velocity(j),
                                   pairwise_drag_ji * old_velocities(i));
              }
            },
            Kokkos::Sum<decltype(delta_i)>(delta_i),
            Kokkos::Sum<Scalar>(drag_i),
            Kokkos::Sum<typename PositionsView::value_type>(vel_i)
          );

          Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
            out_view_pack.apply([&](auto&... views) {
              delta_i.apply([&](auto&... values) {
                ((views(i) += values), ...);
              });
            });
            out_view_pack.first()(i) += vel_i /
              Kokkos::fmax(drag_i, Kokkos::Experimental::epsilon_v<Scalar>);
          });
        }
      );

      Kokkos::parallel_for(
        "naive_gabriel_pr_double_apply_drag_j",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::benchmark

#endif // TECS_BENCHMARK_NAIVE_GABRIEL_PARALLEL_REDUCE_DOUBLE_HPP
