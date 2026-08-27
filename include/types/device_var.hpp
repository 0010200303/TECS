#ifndef TECS_TYPES_DEVICE_VAR_HPP
#define TECS_TYPES_DEVICE_VAR_HPP

#include <Kokkos_Core.hpp>

namespace tecs {

  /**
   * @brief A single value that lives on the device and can be read/written
   *        from both host and device code.
   *
   * DeviceVar wraps a `Kokkos::View<T>` of size 1, so the data stays in
   * device memory while still being accessible from the host via deep
   * copies. It is useful for things like counters, scalars, or flags that
   * need to be updated inside parallel kernels and then read back on the
   * host.
   *
   * @tparam T The underlying scalar or struct type.
   *
   * **Important limitation:**
   * Only **full-value assignment** (`var = value`) is supported — modifying
   * individual fields of a struct on the host (e.g. `my_vector.x = 13`)
   * will **not** be copied back to the device.
   *
   * **Basic usage:**
   * @code
   *   // On the host
   *   DeviceVar<int> counter("my_counter", 0);
   *
   *   // Inside a kernel
   *   int old = Kokkos::atomic_add_fetch(counter.data(), 1);
   *
   *   // Back on the host
   *   int result = counter;   // implicit conversion reads from device
   * @endcode
   */
  template<typename T>
  struct DeviceVar {
    /**
     * @brief Default-construct with a zero-initialized value.
     */
    DeviceVar() : device_view("DeviceVar (default constructed)") {
      Kokkos::deep_copy(device_view, T());
    }
    
    /**
     * @brief Construct with a label and zero-initialized value.
     * @param label Kokkos debug label.
     */
    DeviceVar(const std::string& label) : device_view(label) {
      Kokkos::deep_copy(device_view, T());
    }

    /**
     * @brief Construct with a label and an initial value.
     * @param label Kokkos debug label.
     * @param value Initial value, copied to the device.
     */
    DeviceVar(const std::string& label, const T& value) : device_view(label) {
      Kokkos::deep_copy(device_view, value);
    }

    /**
     * @brief Construct from a value directly (uses a generic label).
     * @param value Initial value, copied to the device.
     */
    DeviceVar(const T& value) : device_view("DeviceVar (directly assigned)") {
      Kokkos::deep_copy(device_view, value);
    }

    /// The underlying 1-element Kokkos View on the device.
    Kokkos::View<T> device_view;

    /**
     * @brief Device-side copy constructor (shallow, views the same data).
     */
    KOKKOS_INLINE_FUNCTION
    DeviceVar(const DeviceVar& other) : device_view(other.device_view) { }

    /**
     * @brief Implicit conversion to the underlying type.
     *
     * - On the **device**: reads the value directly from memory.
     * - On the **host**: performs a deep copy from the device.
     */
    KOKKOS_INLINE_FUNCTION
    operator T() const {
      if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Kokkos::View<T>::memory_space>::accessible) {
        return device_view();
      } else {
        KOKKOS_IF_ON_DEVICE((
          return *device_view.data();
        ))
        KOKKOS_IF_ON_HOST((
          T host_value;
          Kokkos::deep_copy(host_value, device_view);
          return host_value;
        ))
      }
    }

    /**
     * @brief Assign a new value.
     *
     * - On the **device**: writes directly to memory.
     * - On the **host**: performs a deep copy to the device.
     */
    KOKKOS_INLINE_FUNCTION
    DeviceVar& operator=(const T& value) {
      KOKKOS_IF_ON_DEVICE((
        *device_view.data() = value;
      ))
      KOKKOS_IF_ON_HOST((
        Kokkos::deep_copy(device_view, value);
      ))
      return *this;
    }

    /**
     * @brief Return a raw pointer to the device data.
     *
     * Useful for atomic operations inside kernels.
     */
    KOKKOS_INLINE_FUNCTION
    T* data() {
      return device_view.data();
    }

    /// @copydoc data()
    KOKKOS_INLINE_FUNCTION
    T* data() const {
      return device_view.data();
    }
  };
} // namespace tecs

#endif // TECS_TYPES_DEVICE_VAR_HPP
