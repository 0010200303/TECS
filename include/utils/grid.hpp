#ifndef TECS_UTILS_GRID_HPP
#define TECS_UTILS_GRID_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "functions.hpp"
#include "../types/view.hpp"

namespace tecs::acceleration {

  /**
   * @brief A uniform spatial grid (binning) for accelerating spatial queries.
   *
   * The Grid subdivides the simulation domain into a regular lattice of bins
   * and assigns every agent to a bin based on its position. It uses
   * Kokkos::BinSort to produce a permutation vector that groups agents by bin,
   * enabling fast neighbourhood queries without checking all pairs.
   *
   * Typical uses include:
   * - Finding the nearest agent to a given point.
   * - Picking a random agent within a given radius (uniformly distributed).
   * - Accelerating pairwise force evaluations (used internally by binned pair finders).
   *
   * @tparam Vector              The agent position vector type.
   * @tparam PositionsViewType   The Kokkos view type holding positions.
   */
  template<typename Vector, typename PositionsViewType = View<Vector>>
  class Grid {
    EXTRACT_VECTOR(Vector)
    using ViewType = PositionsViewType;
    using VectorI = VectorN<int, dimensions>;
    using BinOp = Kokkos::BinOp1D<Kokkos::View<int*>>;
    using BinSort = Kokkos::BinSort<Kokkos::View<int*>, BinOp>;

    /**
     * @brief Estimate a suitable bin size so that each bin contains roughly
     *        @p agents_per_bin_ agents on average.
     *
     * @return Uniform bin edge length (same in all dimensions).
     */
    static Scalar get_bin_size(
      const Vector& min_bounds_,
      const Vector& max_bounds_,
      const unsigned int agent_count_,
      const unsigned int agents_per_bin_ = 1
    ) {
      if (agent_count_ == 0)
        return Scalar(1);

      Scalar volume = 1;
      for (int d = 0; d < dimensions; ++d)
        volume *= (max_bounds_[d] - min_bounds_[d]);

      return static_cast<Scalar>(std::pow(
        (static_cast<Scalar>(agents_per_bin_) * volume) / static_cast<Scalar>(agent_count_), 1.0 / dimensions
      ));
    }

    public:
      /**
       * @brief Construct a grid with explicit bounds and bin size.
       *
       * @param data_view_   View containing agent positions.
       * @param agent_count_ Number of active agents.
       * @param min_bounds_  Minimum coordinate in each dimension.
       * @param max_bounds_  Maximum coordinate in each dimension.
       * @param bin_size_    Edge length of each (square/cubic) bin.
       */
      Grid(
        const ViewType& data_view_,
        const unsigned int agent_count_,
        const Vector& min_bounds_,
        const Vector& max_bounds_,
        const Scalar bin_size_
      ) : data_view(data_view_)
        , agent_count(agent_count_)
        , min_bounds(min_bounds_)
        , max_bounds(max_bounds_)
        , bin_size(bin_size_)
        , bin_extents(calc_bin_extents())
        , bin_count(calc_bin_count())
        , particle_bins(std::string("Grid") + std::to_string(dimensions) + "_particle_bins", agent_count_)
        , sorter(particle_bins, 0, agent_count_, BinOp{bin_count, 0, bin_count}) { }
      
      /**
       * @brief Construct a grid with a Bounds struct and auto-computed bin size.
       *
       * The bin size is chosen so that each bin contains roughly
       * @p agents_per_bin_ agents.
       */
      Grid(
        const ViewType& data_view_,
        const unsigned int agent_count_,
        const utils::Bounds<Vector>& bounds,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(
          data_view_,
          agent_count_,
          bounds.min,
          bounds.max,
          get_bin_size(bounds.min, bounds.max, agent_count_, agents_per_bin_)) { }

      /**
       * @brief Construct a grid from a view, computing bounds automatically.
       *
       * @copydetails Grid(const ViewType&, unsigned int, const utils::Bounds<Vector>&, unsigned int)
       */
      Grid(
        const ViewType& data_view_,
        const unsigned int agent_count_,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(
          data_view_,
          agent_count_,
          utils::get_bounds<Vector, ViewType>(data_view_),
          agents_per_bin_) { }

      /**
       * @brief Construct a grid from a full view (agent count = view extent).
       */
      Grid(
        const ViewType& data_view_,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(data_view_, data_view_.extent(0), agents_per_bin_) { }

      /// @brief Default constructor (empty grid, zero bins).
      Grid() : sorter(particle_bins, 0, 0, BinOp{0, 0, 0}) {}

    public:
      /// The view containing agent positions.
      ViewType data_view;
      /// Number of active agents.
      unsigned int agent_count = 0;

      /// Lower bound of the grid domain.
      Vector min_bounds;
      /// Upper bound of the grid domain.
      Vector max_bounds;
      /// Edge length of one bin.
      Scalar bin_size = 0.0;

      /// Number of bins along each axis.
      VectorI bin_extents;
      /// Total number of bins (product of all extents).
      int bin_count = 0;

      /// Per-agent bin index (used by the sorter).
      Kokkos::View<int*> particle_bins;
      /// Kokkos BinSort that groups agents by bin index.
      BinSort sorter;

    public:
      /// @brief The permutation vector that maps sorted to original indices.
      KOKKOS_INLINE_FUNCTION
      BinSort::offset_type get_permute_vector() const {
        return sorter.get_permute_vector();
      }

      /// @brief Bin offset array: bin @p i contains indices
      ///        `[offsets[i], offsets[i+1])` in the sorted order.
      KOKKOS_INLINE_FUNCTION
      BinSort::offset_type get_bin_offsets() const {
        return sorter.get_bin_offsets();
      }

      /// @brief Lower bound of the grid domain.
      KOKKOS_INLINE_FUNCTION
      Vector get_min_bounds() const {
        return min_bounds;
      }

      /// @brief Upper bound of the grid domain.
      KOKKOS_INLINE_FUNCTION
      Vector get_max_bounds() const {
        return max_bounds;
      }

      /// @brief Edge length of one bin.
      KOKKOS_INLINE_FUNCTION
      Scalar get_bin_size() const {
        return bin_size;
      }

      /// @brief Number of bins along each axis.
      KOKKOS_INLINE_FUNCTION
      VectorI get_bin_extents() const {
        return bin_extents;
      }

      /// @brief Total number of bins.
      KOKKOS_INLINE_FUNCTION
      int get_bin_count() const {
        return bin_count;
      }

    public:
      /// @brief Compute the number of bins along each axis from bounds and bin size.
      KOKKOS_INLINE_FUNCTION
      constexpr VectorI calc_bin_extents() {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = static_cast<int>(Kokkos::ceil((max_bounds[d] - min_bounds[d]) / bin_size));
        return result;
      }

      /// @brief Compute total bin count from the extents.
      KOKKOS_INLINE_FUNCTION
      constexpr int calc_bin_count() {
        int result = 1;
        for (int d = 0; d < dimensions; ++d)
          result *= bin_extents[d];
        return result;
      }

      /**
       * @brief Compute the integer bin coordinates for a point.
       *
       * The result is clamped to `[0, bin_extents[d] - 1]`.
       */
      KOKKOS_INLINE_FUNCTION
      VectorI calc_bin_coords_from_point(const Vector& point) const {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = Kokkos::max(0, Kokkos::min(
            static_cast<int>((point[d] - min_bounds[d]) / bin_size), bin_extents[d] - 1
          ));
        return result;
      }

      /**
       * @brief Convert multi-dimensional bin coordinates to a flat 1D index.
       */
      KOKKOS_INLINE_FUNCTION
      int flatten_bin_index(const VectorI& coords) const {
        if constexpr (dimensions == 1) {
          return coords[0];
        }
        else if constexpr (dimensions == 2) {
          return coords[0] + coords[1] * bin_extents[0];
        }
        else if constexpr (dimensions == 3) {
          return coords[0] + coords[1] * bin_extents[0] + coords[2] * (bin_extents[0] * bin_extents[1]);
        }
        else if constexpr (dimensions == 4) {
          return coords[0] + coords[1] * bin_extents[0] + coords[2] * (bin_extents[0] * bin_extents[1]) +
            coords[3] * (bin_extents[0] * bin_extents[1] * bin_extents[2]);
        }
        else {
          int idx = 0;
          int stride = 1;
          for (int d = 0; d < dimensions; ++d) {
            idx += coords[d] * stride;
            stride *= bin_extents[d];
          }
          return idx;
        }
      }

      /**
       * @brief Compute the flat bin index for a point in a single call.
       */
      KOKKOS_INLINE_FUNCTION
      int calc_bin_index_from_point(const Vector& point) const {
        return flatten_bin_index(calc_bin_coords_from_point(point));
      }

      /**
       * @brief Number of neighbour-offset combinations for a cube of side @p side.
       *
       * For example, in 2D with side=3 this returns 9 (a 3×3 neighbourhood).
       */
      KOKKOS_INLINE_FUNCTION
      int calc_task_count(const int side) const {
        if constexpr (dimensions == 1) {
          return side;
        }
        else if constexpr (dimensions == 2) {
          return side * side;
        }
        else if constexpr (dimensions == 3) {
          return side * side * side;
        }
        else if constexpr (dimensions == 4) {
          return side * side * side * side;
        }
        else {
          int n = 1;
          for (int d = 0; d < dimensions; ++d)
            n *= side;
          return n;
        }
      }

      /**
       * @brief Convert a linear task index (neighbourhood walk) into a bin offset.
       *
       * Used together with `calc_task_count()` to iterate over all bins in a
       * neighbourhood cube around a centre bin.
       *
       * @param task_idx    Linear index into the neighbourhood (0 ... task_count - 1).
       * @param side        Side length of the neighbourhood cube (2 * radius_bins + 1).
       * @param radius_bins Neighbourhood radius in number of bins.
       * @return An offset vector that can be added to centre bin coordinates.
       */
      KOKKOS_INLINE_FUNCTION
      VectorI linear_index_to_offset(const int task_idx, const int side, const int radius_bins) const {
        VectorI result;
        if constexpr (dimensions == 1) {
          result[0] = task_idx - radius_bins;
        }
        else if constexpr (dimensions == 2) {
          result[0] = (task_idx % side) - radius_bins;
          result[1] = (task_idx / side) - radius_bins;
        }
        else if constexpr (dimensions == 3) {
          const int temp = task_idx / side;
          result[0] = (task_idx % side) - radius_bins;
          result[1] = (temp % side) - radius_bins;
          result[2] = (temp / side) - radius_bins;
        }
        else if constexpr (dimensions == 4) {
          const int temp1 = task_idx / side;
          result[0] = (task_idx % side) - radius_bins;
          result[1] = (temp1 % side) - radius_bins;

          const int temp2 = temp1 / side;
          result[2] = (temp2 % side) - radius_bins;
          result[3] = (temp2 / side) - radius_bins;
        }
        else {
          int temp = task_idx;
          for (int d = 0; d < dimensions; ++d) {
            result[d] = (temp % side) - radius_bins;
            temp /= side;
          }
        }
        return result;
      }

      /**
       * @brief Check whether a bin lies outside the grid extents.
       * @return `true` if the bin is out of bounds.
       */
      KOKKOS_INLINE_FUNCTION
      bool is_bin_outside_extents(const VectorI& bin) const {
        if constexpr (dimensions == 1) {
          return bin[0] < 0 || bin[0] >= bin_extents[0];
        }
        else if constexpr (dimensions == 2) {
          return bin[0] < 0 || bin[0] >= bin_extents[0] ||
                bin[1] < 0 || bin[1] >= bin_extents[1];
        }
        else if constexpr (dimensions == 3) {
          return bin[0] < 0 || bin[0] >= bin_extents[0] ||
                bin[1] < 0 || bin[1] >= bin_extents[1] ||
                bin[2] < 0 || bin[2] >= bin_extents[2];
        }
        else if constexpr (dimensions == 4) {
          return bin[0] < 0 || bin[0] >= bin_extents[0] ||
                bin[1] < 0 || bin[1] >= bin_extents[1] ||
                bin[2] < 0 || bin[2] >= bin_extents[2] ||
                bin[3] < 0 || bin[3] >= bin_extents[3];
        }
        else {
          for (int d = 0; d < dimensions; ++d) {
            if (bin[d] < 0 || bin[d] >= bin_extents[d])
              return true;
          }
          return false;
        }
      }

      /**
       * @brief Number of agents in a given bin (using the offset array).
       */
      KOKKOS_INLINE_FUNCTION
      int get_agent_count_for_bin(const int bin_index) const {
        return get_bin_offsets()(bin_index + 1) - get_bin_offsets()(bin_index);
      }
    
      /**
       * @brief Rebuild the grid (after agents have moved).
       *
       * Recomputes each agent's bin index and re-sorts the permutation vector.
       * Call this whenever positions change significantly.
       */
      void rebuild() {
        Kokkos::parallel_for(
          std::string("Grid") + std::to_string(dimensions) + "rebuild",
          agent_count,
          KOKKOS_CLASS_LAMBDA(const unsigned int i) {
            VectorI bin_coords = calc_bin_coords_from_point(read_element(data_view, i));
            particle_bins(i) = flatten_bin_index(bin_coords);
          }
        );

        sorter = BinSort(particle_bins, 0, agent_count, BinOp{bin_count, 0, bin_count});
        sorter.create_permute_vector();
      }

      /**
       * @brief Find the index of the agent nearest to a given point.
       *
       * Only searches bins within a sector of side `2 * search_radius + 1` centred
       * on the query point's bin.
       *
       * @param point         Query position.
       * @param search_radius Neighbourhood radius in bins (default: 1).
       * @return Index of the nearest agent, or -1 if none found.
       */
      KOKKOS_INLINE_FUNCTION
      int get_nearest_point_index(const Vector& point, const int search_radius = 1) const {
        VectorI center_coords = calc_bin_coords_from_point(point);
        const int side = 2 * search_radius + 1;
        const int task_count = calc_task_count(side);

        int nearest_idx = -1;
        Scalar min_distance_squared = Kokkos::Experimental::finite_max_v<Scalar>;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, search_radius);
          if (is_bin_outside_extents(ni) == true)
            continue;
          
          const int b = flatten_bin_index(ni);
          const unsigned int start = sorter.get_bin_offsets()(b);
          const unsigned int end = sorter.get_bin_offsets()(b + 1);

          for (int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(sorter.get_permute_vector()(idx));

            const Scalar distance_squared = point.distance_to_squared(read_element(data_view, j));
            if (distance_squared < min_distance_squared) {
              nearest_idx = j;
              min_distance_squared = distance_squared;
            }
          }
        }
        return nearest_idx;
      }

      /**
       * @brief Pick a random agent index uniformly from all bins neighbouring
       *        a point (without distance filtering).
       *
       * @tparam Random     Kokkos random generator type.
       * @param point       Query position.
       * @param radius_bins Neighbourhood radius in bins.
       * @param rng         Random number generator.
       * @return Index of the chosen agent, or -1 if none found.
       */
      template<typename Random>
      KOKKOS_INLINE_FUNCTION
      int get_random_point_index_in_neighbourhood_bins(
        const Vector& point,
        const int radius_bins,
        Random& rng
      ) const {
        VectorI center_coords = calc_bin_coords_from_point(point);
        const int side = 2 * radius_bins + 1;
        const int task_count = calc_task_count(side);

        // count total agents in neighbouring bins
        int total = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;
          
          const int b = flatten_bin_index(ni);
          total += get_agent_count_for_bin(b);
        }

        // uniformly pick random point
        int target = rng.rand(0, total);
        int seen = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;
          
          const int b = flatten_bin_index(ni);
          int count = get_agent_count_for_bin(b);
          if (count == 0)
            continue;
          
          if (target < seen + count) {
            unsigned int offset = get_bin_offsets()(b) + target - seen;
            return get_permute_vector()(offset);
          }
          seen += count;
        }
        return -1;
      }

      /**
       * @brief Pick a random agent uniformly from a circular/spherical
       *        neighbourhood around a point.
       *
       * @tparam Random       Kokkos random generator type.
       * @param point         Query position.
       * @param search_radius Physical search radius (same units as positions).
       * @param rng           Random number generator.
       * @return Index of the chosen agent, or -1 if none found.
       */
      template<typename Random>
      KOKKOS_INLINE_FUNCTION
      int get_random_point_index_in_neighbourhood(
        const Vector& point,
        const Scalar search_radius,
        Random& rng
      ) const {
        int radius_bins = Kokkos::max(1, static_cast<int>(Kokkos::ceil(search_radius / bin_size)));
        VectorI center_coords = calc_bin_coords_from_point(point);
        const int side = 2 * radius_bins + 1;
        const int task_count = calc_task_count(side);
        const Scalar search_radius_squared = search_radius * search_radius;

        // count total agents in neighbouring bins
        int total = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;

          const int b = flatten_bin_index(ni);
          const unsigned int start = get_bin_offsets()(b);
          const unsigned int end = get_bin_offsets()(b + 1);

          for (unsigned int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(get_permute_vector()(idx));
            if (point.distance_to_squared(read_element(data_view, j)) <= search_radius_squared)
              ++total;
          }
        }

        if (total == 0)
          return -1;

        // uniformly pick random point
        int target = static_cast<int>(rng.urand(total));
        int seen = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;

          const int b = flatten_bin_index(ni);
          const unsigned int start = get_bin_offsets()(b);
          const unsigned int end = get_bin_offsets()(b + 1);

          for (unsigned int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(get_permute_vector()(idx));
            if (point.distance_to_squared(read_element(data_view, j)) <= search_radius_squared) {
              if (seen == target)
                return j;
              ++seen;
            }
          }
        }

        return -1;
      }

      /**
       * @brief Pick a random agent from a neighbourhood, filtered by a
       *        user-provided filter.
       *
       * @tparam Random       Kokkos random generator type.
       * @tparam FilterFunc   Callable `bool(int index)`.
       * @param point         Query position.
       * @param search_radius Physical search radius.
       * @param rng           Random number generator.
       * @param filter        Only agents for which `filter(index)` returns true
       *                      are eligible.
       * @return Index of the chosen agent, or -1 if none found.
       */
      template<typename Random, typename FilterFunc>
      KOKKOS_INLINE_FUNCTION
      int get_random_point_index_in_neighbourhood(
        const Vector& point,
        const Scalar search_radius,
        Random& rng,
        const FilterFunc& filter
      ) const {
        int radius_bins = Kokkos::max(1, static_cast<int>(Kokkos::ceil(search_radius / bin_size)));
        VectorI center_coords = calc_bin_coords_from_point(point);
        const int side = 2 * radius_bins + 1;
        const int task_count = calc_task_count(side);
        const Scalar search_radius_squared = search_radius * search_radius;

        // count total agents in neighbouring bins
        int total = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;

          const int b = flatten_bin_index(ni);
          const unsigned int start = get_bin_offsets()(b);
          const unsigned int end = get_bin_offsets()(b + 1);

          for (unsigned int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(get_permute_vector()(idx));
            if (point.distance_to_squared(read_element(data_view, j)) <= search_radius_squared && filter(j) == true)
              ++total;
          }
        }

        if (total == 0)
          return -1;

        // uniformly pick random point
        int target = static_cast<int>(rng.urand(total));
        int seen = 0;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, radius_bins);
          if (is_bin_outside_extents(ni))
            continue;

          const int b = flatten_bin_index(ni);
          const unsigned int start = get_bin_offsets()(b);
          const unsigned int end = get_bin_offsets()(b + 1);

          for (unsigned int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(get_permute_vector()(idx));
            if (point.distance_to_squared(read_element(data_view, j)) <= search_radius_squared && filter(j) == true) {
              if (seen == target)
                return j;
              ++seen;
            }
          }
        }

        return -1;
      }
  };
} // namespace tecs::acceleration

#endif // TECS_UTILS_GRID_HPP
