#ifndef TECS_BENCHMARK_NAIVE_GABRIEL_FOR_SYMMETRIC_HPP
#define TECS_BENCHMARK_NAIVE_GABRIEL_FOR_SYMMETRIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabrielForSymmetric {
    using positions_view_type = PositionsView;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    NaiveGabrielForSymmetric(
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

      Kokkos::View<Scalar*> total_drag("for_sym_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("for_sym_total_velocity", agent_count);

      Kokkos::parallel_for(
        "naive_gabriel_for_symmetric_apply_force",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          const auto position_i = input_positions(i);

          for (int j = 0; j < i; ++j) {
            const auto position_j = input_positions(j);
            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();

            if (distance_squared >= cutoff_distance_squared)
              continue;

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
              continue;

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
        }
      );

      Kokkos::parallel_for(
        "naive_gabriel_for_symmetric_apply_drag",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::benchmark

#endif // TECS_BENCHMARK_NAIVE_GABRIEL_FOR_SYMMETRIC_HPP
