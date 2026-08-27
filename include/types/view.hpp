#ifndef TECS_TYPES_VIEW_HPP
#define TECS_TYPES_VIEW_HPP

#include <string>
#include <source_location>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace tecs {

  /**
   * @brief The default View type for simulations - a wrapper around
   *        Kokkos::DualView that tracks modifications and supports a
   *        logical "active count" separate from the capacity.
   *
   * This is the primary array type used throughout the simulation framework.
   * It inherits from `Kokkos::DualView<T*>` so data is accessible on both
   * host and device, and adds:
   *
   * - An **active count** that lets you allocate a large capacity but only
   *   operate on the first N entries (useful for neighbour lists, links, etc.).
   * - Convenience methods `read()` / `write()` / `access()` for explicit
   *   read/write semantics in kernels.
   * - An auto_sync functions that tries to sync in the right direction based
   *   modified flags.
   *
   * @tparam T The element type (e.g. `Vector`, `Link`, `Scalar`).
   *
   * **Basic usage:**
   * @code
   *   // Allocate a View with capacity 1000, initially 100 active entries
   *   tecs::View<int> view("my_data", 100, 1000);
   *
   *   // Inside a kernel:
   *   view(i) = 42;
   *   int val = view(i);
   *   // or explicit
   *   view.write(i) = 42;       // marks the device as modified
   *   int val = view.read(i);   // does not set any flags
   *
   *   // On the host:
   *   view.auto_sync();         // only copies if device was modified
   *   view.set_active_count(n); // grow/shrink the logical size
   * @endcode
   */
  template<typename T>
  struct View : public Kokkos::DualView<T*> {
    /**
     * @brief Default constructor — creates an empty View with capacity 0.
     */
    View()
      : Kokkos::DualView<T*>("tecs::View default", 0)
      , device_modified_flag("DualView:device_modified_flag")
      , active_count(0) { }

    /**
     * @brief Construct a View with a given active count (capacity = active count).
     * @param label         Kokkos debug label.
     * @param active_count_ Number of logically active entries.
     */
    View(const std::string& label, const unsigned int active_count_)
      : Kokkos::DualView<T*>(label, active_count_)
      , device_modified_flag("DualView:device_modified_flag")
      , active_count(active_count_) { }
    
    /**
     * @brief Construct a View with separate active count and capacity.
     *
     * The underlying allocation uses `capacity`, while only the first
     * `active_count_` entries are considered active. This is useful when
     * the maximum number of elements is known but the actual count varies.
     *
     * @param label         Kokkos debug label.
     * @param active_count_ Number of logically active entries.
     * @param capacity      Maximum allocated count.
     */
    View(const std::string& label, const unsigned int active_count_, const unsigned int capacity)
      : Kokkos::DualView<T*>(label, capacity)
      , device_modified_flag("DualView:device_modified_flag")
      , active_count(active_count_) { }

    /**
     * A 1-element View on the device that tracks whether the data was
     * modified inside a kernel (used by `sync_host()`).
     */
    Kokkos::View<bool> device_modified_flag;

    /**
     * Number of logically active entries. The underlying allocation may be
     * larger (see `get_capacity()`). Only entries up to `active_count` are
     * written to HDF5 output.
     */
    unsigned int active_count;

    /// @brief Return the debug label of the underlying device view.
    inline std::string label() const {
      return this->view_device().label();
    }

    /// @brief Views are always 1-dimensional (rank 1).
    KOKKOS_INLINE_FUNCTION
    int rank() const {
      return 1;
    }

    /**
     * @brief Read/write access to element `i`, with automatic flag tracking.
     *
     * When the host and device share the same memory space (UVM, Serial,
     * OpenMP) the modified flags are skipped entirely — the data is always
     * accessible from anywhere.
     *
     * Otherwise:
     * - On the **device**: sets `device_modified_flag` to `true`, so that
     *   a subsequent `sync_host()` knows data needs copying.
     * - On the **host**: marks the host view as modified for DualView's
     *   own sync mechanism.
     */
    KOKKOS_INLINE_FUNCTION
    T& operator()(const int i) const {
      if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Kokkos::DualView<T*>::memory_space>::accessible) {
        return this->view_device()(i);
      } else {
        KOKKOS_IF_ON_DEVICE((
          device_modified_flag() = true;
          return this->view_device()(i);
        ))
        KOKKOS_IF_ON_HOST((
          const_cast<View*>(this)->modify_host();
          return this->view_host()(i);
        ))
      }
    }

    /**
     * @brief Access element `i` **without** setting any modified flag.
     *
     * Use this when you know what you are doing and plan on syncing
     * and plan on syncing manually later without the overhead of 
     * setting the modified flags.
     */
    KOKKOS_INLINE_FUNCTION
    T& access(const int i) const {
      KOKKOS_IF_ON_DEVICE((
        return this->view_device()(i);
      ))
      KOKKOS_IF_ON_HOST((
        return this->view_host()(i);
      ))
    }

    /// @brief Alias for `access()` - explicit read-only access.
    KOKKOS_INLINE_FUNCTION
    T& read(const int i) const {
      return access(i);
    }

    /// @brief Alias for `operator()` - explicit write access.
    KOKKOS_INLINE_FUNCTION
    T& write(const int i) const {
      return (*this)(i);
    }

    /**
     * @brief Synchronise the host view from the device.
     *
     * Checks the `device_modified_flag` first: if a kernel modified the
     * data, it marks the device view as modified and resets the flag,
     * then calls the base-class `sync_host()`. If no kernel modification
     * occurred, only the base-class sync logic applies.
     */
    inline void sync_host() {
      auto flag_host = Kokkos::create_mirror_view(device_modified_flag);
      Kokkos::deep_copy(flag_host, device_modified_flag);

      if (flag_host() == true) {
        this->modify_device();
        Kokkos::deep_copy(device_modified_flag, false);
      }
      Kokkos::DualView<T*>::sync_host();
    }

    /**
     * @brief Force synchronisation from host to device, regardless of flags.
     */
    inline void unconditional_sync_device() {
      this->modify_host();
      Kokkos::DualView<T*>::sync_device();
    }

    /**
     * @brief Force synchronisation from device to host, regardless of flags.
     */
    inline void unconditional_sync_host() {
      this->modify_device();
      Kokkos::deep_copy(device_modified_flag, false);
      Kokkos::DualView<T*>::sync_host();
    }

    /**
     * @brief Automatically sync in the correct direction.
     *
     * - Does nothing if neither view was modified.
     * - Aborts (via DualView) if both views were modified.
     * - Otherwise syncs the modified side to the other.
     */
    inline void auto_sync() {
      bool flag_host;
      Kokkos::deep_copy(flag_host, device_modified_flag);
      if (flag_host == true)
        this->modify_device();

      if (this->need_sync_host())
        this->sync_host();
      else if (this->need_sync_device())
        this->sync_device();
    }

    /**
     * @brief Deep copy from another View.
     *
     * Copies both the data and resets the device-modified flag.
     */
    inline void deep_copy(const View<T>& src) {
      Kokkos::deep_copy(static_cast<Kokkos::DualView<T*>&>(*this), static_cast<const Kokkos::DualView<T*>&>(src));
      Kokkos::deep_copy(device_modified_flag, false);
    }

    /**
     * @brief Fill the entire View with a single value.
     *
     * Sets both host and device copies and clears the modified flag.
     */
    inline void deep_copy(const T& src) {
      Kokkos::deep_copy(this->view_device(), src);
      Kokkos::deep_copy(this->view_host(), src);
      Kokkos::deep_copy(device_modified_flag, false);
    }

    /**
     * @brief Resize the View, resetting the modified flag and syncing host.
     * @param value New size (capacity).
     */
    inline void resize(const int value) {
      Kokkos::DualView<T*>::resize(value);
      Kokkos::deep_copy(device_modified_flag, false);
      Kokkos::DualView<T*>::sync_host();
    }

    /// @brief Return the total allocated capacity (may be > active count).
    KOKKOS_INLINE_FUNCTION
    unsigned int get_capacity() const {
      return this->extent(0);
    }

    /// @brief Return the number of logically active entries.
    inline  unsigned int get_active_count() const {
      return active_count;
    }

    /// @brief Set the number of logically active entries.
    inline void set_active_count(const unsigned int value) {
      active_count = value;
    }
  };
} // namespace tecs

#endif // TECS_TYPES_VIEW_HPP
