#ifndef TECS_BENCHMARK_BINNED_ALL_PAIRS_REDUCE_PARALLEL_DOUBLE_HPP
#define TECS_BENCHMARK_BINNED_ALL_PAIRS_REDUCE_PARALLEL_DOUBLE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"
#include "../../include/utils/grid.hpp"

namespace tecs::benchmark {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct BinnedAllPairsReduceParallelDouble {
    using positions_view_type = PositionsView;
    using Vector = VectorN<Scalar, dimensions>;
    using VectorI = VectorN<int, dimensions>;
    using Grid = acceleration::Grid<Vector, PositionsView>;

    struct Settings {
      Vector min_bounds = Vector(-20);
      Vector max_bounds = Vector( 20);
      Scalar bin_size_scale = Scalar(1);
      int rebuild_every_n = 0;
    };

    BinnedAllPairsReduceParallelDouble(
      unsigned int agent_count_,
      const Scalar cutoff_distance_,
      const Settings& settings)
      : agent_count(agent_count_)
      , cutoff_distance(cutoff_distance_)
      , cutoff_distance_squared(cutoff_distance_ * cutoff_distance_)
      , min_bounds(settings.min_bounds)
      , max_bounds(settings.max_bounds)
      , bin_size(settings.bin_size_scale * cutoff_distance)
      , inv_bin_size(Scalar(1) / bin_size)
      , search_radius(static_cast<int>(Kokkos::ceil(cutoff_distance_ * inv_bin_size)))
      , rebuild_every_n(settings.rebuild_every_n) { }

    unsigned int agent_count;
    const Scalar cutoff_distance;
    const Scalar cutoff_distance_squared;

    const Vector min_bounds;
    const Vector max_bounds;
    const Scalar bin_size;
    const Scalar inv_bin_size;
    const int search_radius;

    Grid grid;

    int step_count = 0;
    int rebuild_every_n = 0;

    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
      step_count = 0;
    }

    void rebuild(const PositionsView& input_positions) {
      grid = Grid(input_positions, agent_count, min_bounds, max_bounds, bin_size);
      grid.rebuild();
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

      // rebuild grid
      if (is_full_step == true) {
        if (step_count == 0 || step_count >= rebuild_every_n) {
          rebuild(input_positions);
          step_count = 0;
        }
        step_count++;
      }

      const int side = 2 * search_radius + 1;
      const int task_count = grid.calc_task_count(side);
      const VectorI bin_extents = grid.get_bin_extents();

      Kokkos::View<Scalar*> total_drag("bapd_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("bapd_total_velocity", agent_count);

      Kokkos::parallel_for(
        "binned_all_pairs_rp_double_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);
          VectorI bin_coords = grid.calc_bin_coords_from_point(position_i);

          auto delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar drag_i = Scalar(0);
          typename PositionsView::value_type vel_i{0};

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, task_count),
            [&](const int task_idx, auto& local_delta_i, auto& local_drag_i, auto& local_vel_i) {
              const VectorI ni = bin_coords + grid.linear_index_to_offset(task_idx, side, search_radius);
              if (grid.is_bin_outside_extents(ni) == true)
                return;

              const int b = grid.flatten_bin_index(ni);
              const unsigned int start = grid.get_bin_offsets()(b);
              const unsigned int end = grid.get_bin_offsets()(b + 1);

              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, start, end),
                [&](const int idx) {
                  const int j = static_cast<int>(grid.get_permute_vector()(idx));
                  if (j == i)
                    return;

                  const auto& position_j = input_positions(j);
                  const auto displacement = position_i - position_j;
                  const auto distance_squared = displacement.length_squared();
                  if (distance_squared >= cutoff_distance_squared)
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
                }
              );
            },
            Kokkos::Sum<decltype(delta_i)>(delta_i),
            Kokkos::Sum<Scalar>(drag_i),
            Kokkos::Sum<typename PositionsView::value_type>(vel_i)
          );

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              out_view_pack.apply([&](auto&... views) {
                delta_i.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });

              out_view_pack.first()(i) += vel_i /
                Kokkos::fmax(drag_i, Kokkos::Experimental::epsilon_v<Scalar>);
            }
          );
        }
      );

      Kokkos::parallel_for(
        "binned_all_pairs_rp_double_apply_drag_j",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::benchmark

#endif // TECS_BENCHMARK_BINNED_ALL_PAIRS_REDUCE_PARALLEL_DOUBLE_HPP
