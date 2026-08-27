#ifndef TECS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP
#define TECS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace tecs::pair_finders {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveDelaunay {
    using positions_view_type = PositionsView;

    struct Settings { };

    static_assert(dimensions >= 2 && dimensions <= 3, "NaiveDelaunay requires 2 or 3 dimensions");

    NaiveDelaunay(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings
    ) : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;

    // benchmark different loops
    template<typename RandomPool, typename Force, typename ForceFields, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force,
      bool is_full_step
    ) {
      using Vector = VectorN<Scalar, dimensions>;

      const auto& input_positions = in_view_pack.first();

      // using delta_type = decltype(detail::make_accumulator_pack(out_view_pack));
      // Kokkos::View<delta_type*> total_delta_view("naive_delaunay_apply_force_total_delta", agent_count);
      Kokkos::View<Scalar*> total_drag_view("naive_delaunay_apply_force_total_drag", agent_count);
      Kokkos::View<Vector*> total_velocity_view("naive_delaunay_apply_force_total_velocity", agent_count);

      if constexpr (dimensions == 2) {
        const unsigned long long block = static_cast<unsigned long long>(agent_count - 1) * (agent_count - 2);
        Kokkos::parallel_for(
          "naive_delaunay_apply_force",
          Kokkos::RangePolicy<Kokkos::IndexType<unsigned long long>>(0, static_cast<unsigned long long>(agent_count) * block),
          KOKKOS_CLASS_LAMBDA(const unsigned long long idx) {
            const unsigned long long rem = idx % block;
            const unsigned int i = static_cast<unsigned int>(idx / block);
            unsigned int j = static_cast<unsigned int>(rem / (agent_count - 2));
            unsigned int k = static_cast<unsigned int>(rem % (agent_count - 2));

            j += j >= i;
            k += k >= (i < j ? i : j);
            k += k >= (i < j ? j : i);

            const auto position_i = input_positions(i);
            const auto position_j = input_positions(j);

            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();
            if (distance_squared >= cutoff_distance_squared)
              return;

            const auto position_k = input_positions(k);

            const Scalar denominator = Scalar(2) * (
              position_i[0] * (position_j[1] - position_k[1]) +
              position_j[0] * (position_k[1] - position_i[1]) +
              position_k[0] * (position_i[1] - position_j[1])
            );

            // collinearity check
            if (Kokkos::abs(denominator) <= Kokkos::Experimental::epsilon_v<Scalar>)
              return;

            Vector circumcenter = Vector(
              (
                position_i.length_squared() * (position_j[1] - position_k[1]) +
                position_j.length_squared() * (position_k[1] - position_i[1]) +
                position_k.length_squared() * (position_i[1] - position_j[1])
              ) / denominator,
              (
                position_i.length_squared() * (position_k[0] - position_j[0]) +
                position_j.length_squared() * (position_i[0] - position_k[0]) +
                position_k.length_squared() * (position_j[0] - position_i[0])
              ) / denominator
            );
            const Scalar radius_squared = position_i.distance_to_squared(circumcenter);

            // check every other cell
            for (int m = 0; m < agent_count; ++m) {
              if (m == i || m == j || m == k)
                continue;

              if (input_positions(m).distance_to_squared(circumcenter) <= radius_squared)
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

            // total_delta_view(i).apply([&](auto&... total_deltas) {
              // local_delta.apply([&](auto&... local_deltas) {
                // ((Kokkos::atomic_add(&total_deltas, local_deltas)), ...);
              // });
            // });

            out_view_pack.apply([&](auto&... views) {
              local_delta.apply([&](auto&... local_deltas) {
                ((Kokkos::atomic_add(&views(i), local_deltas)), ...);
              });
            });

            Kokkos::atomic_add(&total_drag_view(i), pairwise_drag);
            Kokkos::atomic_add(&total_velocity_view(i), pairwise_drag * old_velocities(j));

            return;
          }
        );

        Kokkos::parallel_for(
          "naive_delaunay_apply_changes",
          agent_count,
          KOKKOS_CLASS_LAMBDA(const int i) {
            // out_view_pack.apply([&](auto&... views) {
              // total_delta_view(i).apply([&](auto&... values) {
                // ((views(i) += values), ...);
              // });
            // });

            out_view_pack.first()(i) += total_velocity_view(i) /
              Kokkos::fmax(total_drag_view(i), Kokkos::Experimental::epsilon_v<Scalar>);
          }
        );
      }
      else if constexpr (dimensions == 3) {
        Kokkos::parallel_for(
          "naive_delaunay_apply_force",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {agent_count, agent_count - 1, agent_count}),
          KOKKOS_CLASS_LAMBDA(const unsigned int i, unsigned int j, unsigned int k) {
            j += j >= i;

            // Skip i and j so that k is distinct from both.  We shift k past
            // the lower excluded index first, then past the higher, which
            // guarantees the final k is in [0, agent_count) and k != i, k != j.
            const unsigned int lo = (i < j) ? i : j;
            const unsigned int hi = (i < j) ? j : i;
            k += k >= lo;
            k += k >= hi;

            // k may now equal agent_count (when i,j are the two smallest
            // indices); the inner l-loop won't find a valid l for such k, so
            // it's harmless but wastes a tiny amount of work.
            if (k >= agent_count) return;

            const auto position_i = input_positions(i);
            const auto position_j = input_positions(j);

            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();
            if (distance_squared >= cutoff_distance_squared)
              return;

            const auto position_k = input_positions(k);

            // inner loop over l > k, skipping i, j, k
            for (unsigned int l = 0; l < agent_count; ++l) {
              if (l == i || l == j || l == k || l <= k) continue;

              const auto position_l = input_positions(l);

              const Vector edge1 = position_j - position_i;
              const Vector edge2 = position_k - position_i;
              const Vector edge3 = position_l - position_i;

              const Scalar denominator = Scalar(2) * (
                edge1[0] * (edge2[1] * edge3[2] - edge2[2] * edge3[1]) -
                edge1[1] * (edge2[0] * edge3[2] - edge2[2] * edge3[0]) +
                edge1[2] * (edge2[0] * edge3[1] - edge2[1] * edge3[0])
              );

              if (Kokkos::abs(denominator) <= Kokkos::Experimental::epsilon_v<Scalar>)
                continue;

              const Scalar b1 = position_j.length_squared() - position_i.length_squared();
              const Scalar b2 = position_k.length_squared() - position_i.length_squared();
              const Scalar b3 = position_l.length_squared() - position_i.length_squared();

              Vector circumcenter = Vector(
                position_i[0] + (
                  b1       * (edge2[1] * edge3[2] - edge2[2] * edge3[1]) -
                  edge1[1] * (b2       * edge3[2] - edge2[2] * b3)       +
                  edge1[2] * (b2       * edge3[1] - edge2[1] * b3)
                ) / denominator,
                position_i[1] + (
                  edge1[0] * (b2       * edge3[2] - edge2[2] * b3)       -
                  b1       * (edge2[0] * edge3[2] - edge2[2] * edge3[0]) +
                  edge1[2] * (edge2[0] * b3       - b2       * edge3[0])
                ) / denominator,
                position_i[2] + (
                  edge1[0] * (edge2[1] * b3       - b2       * edge3[1]) -
                  edge1[1] * (edge2[0] * b3       - b2       * edge3[0]) +
                  b1       * (edge2[0] * edge3[1] - edge2[1] * edge3[0])
                ) / denominator
              );
              const Scalar radius_squared = position_i.distance_to_squared(circumcenter);

              // bounding-box prefilter for circumsphere emptiness check
              const Scalar cx = circumcenter[0];
              const Scalar cy = circumcenter[1];
              const Scalar cz = circumcenter[2];
              const Scalar radius = Kokkos::sqrt(radius_squared);

              bool valid = true;
              for (unsigned int m = 0; m < agent_count; ++m) {
                if (m == i || m == j || m == k || m == l) continue;
                const auto pm = input_positions(m);

                if (Kokkos::abs(pm[0] - cx) > radius ||
                    Kokkos::abs(pm[1] - cy) > radius ||
                    Kokkos::abs(pm[2] - cz) > radius)
                  continue;

                if (pm.distance_to_squared(circumcenter) <= radius_squared) {
                  valid = false;
                  break;
                }
              }
              if (valid == false)
                continue;

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

              // Edge (i,j) is Delaunay as soon as ANY l forms a valid
              // circumsphere.  Short-circuit the l-loop after the first hit.
              return;
            }
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
    }
  };
} // namespace tecs::pair_finders

#endif // TECS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP
