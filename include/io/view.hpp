#ifndef TECS_IO_VIEW_HPP
#define TECS_IO_VIEW_HPP

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>

namespace HighFive::details {

  /**
   * @brief HighFive inspector that enables reading/writing Kokkos::View
   *        objects directly from/to HDF5 datasets.
   *
   * This specialization tells the HighFive library how to handle
   * Kokkos::View types so that HDF5_Reader and HDF5_Writer can use 
   * `dataset.read(view)` and `group.createDataSet(label, view)`
   * without extra copies.
   *
   * @note Only Kokkos::Views whose memory is accessible from the host
   *       space are supported (enforced by a static_assert).
   */
  template <typename DataType, typename... Properties>
  struct inspector<Kokkos::View<DataType, Properties...>> {
    using type = Kokkos::View<DataType, Properties...>;
    
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename type::memory_space>::accessible,
      "HighFive can only operate on Kokkos Views accessible from the host memory space");

    using value_type = typename type::value_type;
    using base_type = typename inspector<value_type>::base_type;
    using hdf5_type = typename inspector<value_type>::hdf5_type;

    static constexpr size_t ndim = inspector<value_type>::ndim + type::rank;
    static constexpr size_t min_ndim = inspector<value_type>::min_ndim + type::rank;
    static constexpr size_t max_ndim = inspector<value_type>::max_ndim + type::rank;
    static constexpr bool is_trivially_nestable = inspector<value_type>::is_trivially_nestable;
    static constexpr bool is_trivially_copyable = inspector<value_type>::is_trivially_copyable;

    /// @brief Compute the total rank, including inner dimensions.
    static size_t getRank(const type& value) {
      if (value.size() == 0)
        return type::rank + inspector<value_type>::ndim;
      return type::rank + inspector<value_type>::getRank(value.data()[0]);
    }

    /**
     * @brief Return the extent of each dimension, flattening nested views.
     *
     * For a `Kokkos::View<Kokkos::View<float*>*>` with shape [N][M],
     * this returns `{N, M}`.
     */
    static std::vector<size_t> getDimensions(const type& value) {
      std::vector<size_t> result;
      for (size_t i = 0; i < type::rank; ++i)
        result.push_back(value.extent(i));

      if (value.size() > 0) {
        auto inner_dim = inspector<value_type>::getDimensions(value.data()[0]);
        result.insert(result.end(), inner_dim.begin(), inner_dim.end());
      }
      else {
        std::vector<size_t> empty(inspector<value_type>::ndim, 0);
        result.insert(result.end(), empty.begin(), empty.end());
      }
      return result;
    }

    /// @brief No-op; views must be pre-allocated by the user.
    static void prepare(type& value, const std::vector<size_t>& next_dims) { }

    /// @brief Return a pointer to the underlying HDF5-compatible data.
    static hdf5_type* data(type& value) {
      return inspector<value_type>::data(value.data()[0]);
    }

    /// @copydoc data(type&)
    static const hdf5_type* data(const type& value) {
      return inspector<value_type>::data(value.data()[0]);
    }

    /**
     * @brief Flatten a nested Kokkos::View into a contiguous HDF5 buffer.
     *
     * Recursively serializes all elements, handling nested views.
     */
    static void serialize(const type& val, const std::vector<size_t>& m_dims, hdf5_type* m) {
      auto* ptr = val.data();
      std::vector<size_t> next_dims(m_dims.begin() + type::rank, m_dims.end());
      
      size_t next_size = 1;
      for (auto d : next_dims)
        next_size *= d;
      
      for (size_t i = 0; i < val.size(); ++i)
        inspector<value_type>::serialize(ptr[i], next_dims, m + i * next_size);
    }

    /**
     * @brief Reconstruct a nested Kokkos::View from a contiguous HDF5 buffer.
     *
     * Inverse of serialize().
     */
    static void unserialize(type& val, const std::vector<size_t>& m_dims, const hdf5_type* m) {
      auto* ptr = val.data();
      std::vector<size_t> next_dims(m_dims.begin() + type::rank, m_dims.end());
      
      size_t next_size = 1;
      for (auto d : next_dims)
        next_size *= d;
      
      for (size_t i = 0; i < val.size(); ++i)
        inspector<value_type>::unserialize(ptr[i], next_dims, m + i * next_size);
    }
  };
} // namespace HighFive::details

#endif // TECS_IO_VIEW_HPP
