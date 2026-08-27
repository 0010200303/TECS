// https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004952

#ifndef TECS_TYPES_LINK_HPP
#define TECS_TYPES_LINK_HPP

namespace tecs {

  /**
   * @brief Represents a connection (edge / link / bond) between two agents.
   *
   * Links are used throughout the framework to model neighbour relationships,
   * cell-cell adhesion, spring forces, or any pairwise interaction between
   * agents. Each link stores the indices of the two connected agents.
   *
   * \n [PLOS Comp. Biol. 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004952)
   */
  struct Link {
    /// Links are always pairwise (two indices), so the "dimension" is 2.
    static constexpr unsigned int dimensions = 2;

    /// Index of the first connected agent.
    unsigned int a;
    /// Index of the second connected agent.
    unsigned int b;
  };
} // namespace tecs



namespace HighFive::details {

  /**
   * @brief HighFive inspector that enables HDF5 read/write for Link.
   *
   * Serialises a `Link` as a rank-1 dataset of two unsigned integers
   * (the two agent indices), so links are stored compactly in HDF5 files.
   */
  template<>
  struct inspector<tecs::Link> {
    using type = tecs::Link;
    using base_type = unsigned int;
    using hdf5_type = typename inspector<unsigned int>::hdf5_type;

    static constexpr size_t ndim = 1;
    static constexpr size_t min_ndim = 1;
    static constexpr size_t max_ndim = 1;
    static constexpr bool is_trivially_nestable = false;
    static constexpr bool is_trivially_copyable = std::is_trivially_copyable<type>::value;

    /// @brief A Link is stored as a 1-D array (rank 1).
    static size_t getRank(const type&) { return ndim; }

    /// @brief A Link has two components (`a` and `b`).
    static std::vector<size_t> getDimensions(const type&) {
      return {2};
    }

    /// @brief No-op; the user pre-allocates the output View.
    static void prepare(type&, const std::vector<size_t>&) { }

    /// @brief Return a pointer to the first component (`a`).
    static hdf5_type* data(type& value) {
      return &value.a;
    }

    /// @copydoc data(type&)
    static const hdf5_type* data(const type& value) {
      return &value.a;
    }

    /// @brief Flatten a Link into a two-element buffer `[a, b]`.
    static void serialize(const type& val, const std::vector<size_t>& m_dims, hdf5_type* m) {
      (void)m_dims;
      m[0] = val.a;
      m[1] = val.b;
    }

    /// @brief Reconstruct a Link from a two-element buffer `[a, b]`.
    static void unserialize(const hdf5_type* vec, const std::vector<size_t>& m_dims, type& val) {
      (void)m_dims;
      val.a = vec[0];
      val.b = vec[1];
    }
  };
} // namespace HighFive::details

#endif // TECS_TYPES_LINK_HPP
