#ifndef TECS_BENCHMARK_NAIVE_GABRIEL_SPREAD_HPP
#define TECS_BENCHMARK_NAIVE_GABRIEL_SPREAD_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveGabrielSpread {
    using positions_view_type = PositionsView;
    using Vector = VectorN<Scalar, dimensions>;

    struct Settings {
      Scalar gabriel_coefficient = Scalar(0.8);
    };

    NaiveGabrielSpread(
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

      Kokkos::View<Scalar*> total_drag_view("naive_delaunay_apply_force_total_drag", agent_count);
      Kokkos::View<Vector*> total_velocity_view("naive_delaunay_apply_force_total_velocity", agent_count);

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {agent_count, agent_count}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
          if (i == j)
            return;

          const auto position_i = input_positions(i);
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

          auto local_delta = detail::make_accumulator_pack(out_view_pack);
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

          out_view_pack.apply([&](auto&... views) {
            local_delta.apply([&](auto&... local_deltas) {
              ((Kokkos::atomic_add(&views(i), local_deltas)), ...);
            });
          });

          Kokkos::atomic_add(&total_drag_view(i), pairwise_drag);
          Kokkos::atomic_add(&total_velocity_view(i), pairwise_drag * old_velocities(j));
        }
      );

      Kokkos::parallel_for(
        "naive_delaunay_apply_changes",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity_view(i) /
            Kokkos::fmax(total_drag_view(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_BENCHMARK_NAIVE_GABRIEL_SPREAD_HPP
