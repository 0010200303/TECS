#ifndef TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_SYMMETRIC_HPP
#define TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_SYMMETRIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"
#include "../utils/grid.hpp"

namespace tecs::pair_finders {

  /**
   * @brief Binned all-pairs pair finder with symmetric force evaluation.
   *
   * Organises particles into a regular grid of bins with size proportional to
   * the cutoff distance.  For each particle @f$i@f$, only the neighbouring
   * bins within the search radius are visited.  Within those bins, only
   * particles with index @f$j < i@f$ are considered, so that each unordered
   * pair is evaluated exactly once.
   *
   * Force deltas are computed once and atomically applied to both @f$i@f$
   * (add) and @f$j@f$ (subtract).  Drag and velocity contributions are
   * accumulated with atomics and normalised in a separate finalisation pass.
   *
   * Compared to the non-symmetric BinnedAllPairs this halves the number of
   * pairwise evaluations while preserving the spatial binning acceleration.
   * The atomics introduce some contention, but the approach remains very
   * effective when the cutoff radius is small relative to the domain size.
   *
   * @tparam PositionsView Kokkos view type for particle positions.
   * @tparam Scalar         Floating-point type (e.g., \c double).
   * @tparam dimensions     Number of spatial dimensions (1, 2, or 3).
   */
  template<typename PositionsView, typename Scalar, int dimensions>
  struct BinnedAllPairsSymmetric {
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

    /**
     * @brief Constructs the binned symmetric all-pairs pair finder.
     * @param agent_count_     Number of particles.
     * @param cutoff_distance_ Interaction cutoff distance.
     * @param settings         Binning configuration (bounds, bin scale, rebuild
     *                         period).
     */
    BinnedAllPairsSymmetric(
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

    /** @brief Updates the number of particles and resets the rebuild counter. */
    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
      step_count = 0;
    }

    /** @brief Rebuilds the spatial binning grid from current positions. */
    void rebuild(const PositionsView& input_positions) {
      grid = Grid(input_positions, agent_count, min_bounds, max_bounds, bin_size);
      grid.rebuild();
    }

    /**
     * @brief Evaluates forces for all symmetric pairs using binned
     *        spatial acceleration.
     *
     * The grid is rebuilt at the start of each full step (or according to
     * \c rebuild_every_n).  Each particle @f$i@f$ walks its neighbouring
     * bins, and within each bin only particles with @f$j < i@f$ are
     * considered.  The user force functor is invoked once per pair.
     * Deltas, drag, and velocity corrections are accumulated atomically
     * for both @f$i@f$ and @f$j@f$.  A final kernel normalises the drag
     * correction for all particles.
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

      Kokkos::View<Scalar*> total_drag("bap_sym_total_drag", agent_count);
      Kokkos::View<typename PositionsView::value_type*>
        total_velocity("bap_sym_total_velocity", agent_count);

      Kokkos::parallel_for(
        "binned_all_pairs_symmetric_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);
          VectorI bin_coords = grid.calc_bin_coords_from_point(position_i);

          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, task_count),
            [&](const int task_idx) {
              // map task_idx -> per-dimension offsets in [-search_radius, search_radius]
              const VectorI ni = bin_coords + grid.linear_index_to_offset(task_idx, side, search_radius);
              if (grid.is_bin_outside_extents(ni) == true)
                return;

              const int b = grid.flatten_bin_index(ni);
              const unsigned int start = grid.get_bin_offsets()(b);
              const unsigned int end = grid.get_bin_offsets()(b + 1);

              for (unsigned int idx = start; idx < end; ++idx) {
                const int j = static_cast<int>(grid.get_permute_vector()(idx));
                if (j >= i)
                  continue;

                const auto& position_j = input_positions(j);
                const auto displacement = position_i - position_j;
                const auto distance_squared = displacement.length_squared();
                if (distance_squared >= cutoff_distance_squared)
                  continue;

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
            }
          );
        }
      );

      Kokkos::parallel_for(
        "binned_all_pairs_symmetric_apply_drag",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const int i) {
          out_view_pack.first()(i) += total_velocity(i) /
            Kokkos::fmax(total_drag(i), Kokkos::Experimental::epsilon_v<Scalar>);
        }
      );
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_BINNED_ALL_PAIRS_SYMMETRIC_HPP
