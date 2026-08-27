#ifndef TECS_UTILS_RUNTIME_GUARD_HPP
#define TECS_UTILS_RUNTIME_GUARD_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <mutex>

namespace tecs::detail {

  /**
   * @brief RAII guard that manages Kokkos initialization and finalization.
   *
   * Used internally by the Simulation class. It ensures that `Kokkos::initialize()`
   * is called before any Kokkos kernels run and that `Kokkos::finalize()` is
   * called once all Simulation objects have gone out of scope.
   *
   * The guard uses a reference count so that multiple Simulation objects can
   * coexist safely - only the first construction triggers initialization and
   * only the last destruction triggers finalization. A mutex protects the
   * reference count for thread-safe construction.
   *
   * If Kokkos was already initialized by the user (or another library) before
   * the first guard is created, the guard will **not** call `finalize()` -
   * it only finalizes what it initialized itself.
   */
  class RuntimeGuard {
    public:
      /// @brief Acquire a reference; initialize Kokkos if this is the first.
      RuntimeGuard() { acquire(); }

      /// @brief Release a reference; finalize Kokkos if this was the last
      ///        and the guard owns the runtime.
      ~RuntimeGuard() noexcept { release(); }

      RuntimeGuard(const RuntimeGuard&) = delete;
      RuntimeGuard& operator=(const RuntimeGuard&) = delete;
      RuntimeGuard(RuntimeGuard&&) = delete;
      RuntimeGuard& operator=(RuntimeGuard&&) = delete;

    private:
      /// @brief Mutex protecting the reference count (thread-safe).
      static std::mutex& guard_mutex() {
        static std::mutex mtx;
        return mtx;
      }

      /// @brief Number of active RuntimeGuard instances.
      static std::size_t& ref_count() {
        static std::size_t count = 0;
        return count;
      }

      /// @brief Whether this guard (not an external caller) initialized Kokkos.
      static bool& owns_runtime() {
        static bool owns = false;
        return owns;
      }

      /**
       * @brief Increment the reference count and initialize Kokkos if needed.
       *
       * If the reference count was zero and Kokkos is not yet initialized,
       * calls `Kokkos::initialize()` and marks the runtime as owned.
       */
      static void acquire() {
        std::lock_guard<std::mutex> lock(guard_mutex());

        if (ref_count() == 0) {
          if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            owns_runtime() = true;
          } else {
            owns_runtime() = false;
          }
        }

        ++ref_count();
      }

      /**
       * @brief Decrement the reference count and finalize Kokkos if appropriate.
       *
       * Only calls `Kokkos::finalize()` when the count reaches zero **and**
       * this guard owns the runtime.
       */
      static void release() noexcept {
        std::lock_guard<std::mutex> lock(guard_mutex());

        if (ref_count() == 0)
          return;
        --ref_count();

        if (ref_count() == 0 && owns_runtime()) {
          if (Kokkos::is_initialized())
            Kokkos::finalize();
          owns_runtime() = false;
        }
      }
  };
} // namespace tecs::detail

#endif // TECS_UTILS_RUNTIME_GUARD_HPP
