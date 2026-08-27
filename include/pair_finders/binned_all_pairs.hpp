#ifndef TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_HPP
#define TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"
#include "../utils/grid.hpp"

namespace tecs::pair_finders {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct BinnedAllPairs {
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

    BinnedAllPairs(
      unsigned int agent_count_,
      Scalar cutoff_distance_,
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

      const auto local_permute = grid.get_permute_vector();
      const auto local_offsets = grid.get_bin_offsets();
      const auto local_bin_extents = grid.get_bin_extents();
      const auto local_min = min_bounds;
      const int local_search_radius = search_radius;
      const Scalar local_inv_bin = inv_bin_size;
      const Scalar local_cutoff_squared = cutoff_distance_squared;

      Kokkos::parallel_for(
        "binned_all_pairs_apply_force",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          const auto& position_i = input_positions(i);

          VectorI bin_coords;
          for (int d = 0; d < dimensions; ++d) {
            bin_coords[d] = Kokkos::max(0, Kokkos::min(
              static_cast<int>((position_i[d] - local_min[d]) * local_inv_bin),
              local_bin_extents[d] - 1
            ));
          }

          // setup data for accumulation
          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_drag_i = Scalar(0);
          typename PositionsView::value_type total_velocity_i{0};

          // walk neighbour bins
          for (int t = 0; t < task_count; ++t) {
            int temp = t;
            VectorI neighbour_bin;
            for (int d = 0; d < dimensions; ++d) {
              neighbour_bin[d] = bin_coords[d] + (temp % side) - local_search_radius;
              temp /= side;
            }

            bool out = false;
            for (int d = 0; d < dimensions && !out; ++d)
              out = (neighbour_bin[d] < 0 || neighbour_bin[d] >= local_bin_extents[d]);
            if (out)
              continue;

            int bi = neighbour_bin[0];
            {
              int stride = 1;
              for (int d = 1; d < dimensions; ++d) {
                stride *= local_bin_extents[d - 1];
                bi += neighbour_bin[d] * stride;
              }
            }

            const unsigned int start = local_offsets(bi);
            const unsigned int end = local_offsets(bi + 1);
            for (unsigned int idx = start; idx < end; ++idx) {
              const int j = static_cast<int>(local_permute(idx));
              if (j == i)
                continue;

              const auto& position_j = input_positions(j);
              const auto displacement = position_i - position_j;
              const auto distance_squared = displacement.length_squared();
              if (distance_squared >= local_cutoff_squared)
                continue;

              const auto distance = Kokkos::sqrt(distance_squared);

              Scalar pairwise_drag = Scalar(0);

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
          }

          // commit deltas
          out_view_pack.apply([&](auto&... views) {
            total_delta_i.apply([&](auto&... values) {
              ((views(i) += values), ...);
            });
          });

          out_view_pack.first()(i) += total_velocity_i / Kokkos::fmax(total_drag_i, Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_HPP
