#ifndef TECS_IO_DUMMY_HPP
#define TECS_IO_DUMMY_HPP

#include <Kokkos_Core.hpp>

#include "../types/view.hpp"
#include "../types/link.hpp"

namespace tecs::io {

  /**
   * @brief No-op output class that discards all simulation data.
   *
   * Dummy implements the same interface as HDF5_Writer but performs no
   * writes. It is useful for benchmarking or testing: you can swap it in
   * to measure simulation performance without any I/O overhead while
   * keeping the same calling code.
   *
   * @tparam SimulationConfig A simulation configuration struct (matched
   *         here only for API compatibility; it is not used).
   */
  template<typename SimulationConfig>
  class Dummy {
    public:
      /**
       * @brief Placeholder settings (all fields accept defaults).
       */
      struct Settings { };

      /**
       * @brief Construct a Dummy writer.
       *
       * @param path      Ignored; accepted for API compatibility.
       * @param settings  Ignored; accepted for API compatibility.
       */
      Dummy(const std::string& path, const Settings& settings) { }

      /**
       * @brief Discard one time step of simulation data (no-op).
       *
       * @param time        Simulation time (ignored).
       * @param step        Step index (ignored).
       * @param first_view  Position View (ignored).
       * @param rest_views  Additional Views (ignored).
       */
      template<typename T0, typename... Ts>
      void write(const double time, const unsigned int step, View<T0>& first_view, View<Ts>&... rest_views) { }

      /**
       * @brief Discard static simulation data (no-op).
       *
       * @param static_views One or more Views to ignore.
       */
      template<typename... Ts>
      void write_static(View<Ts>&... static_views) { }
  };
} // namespace tecs::io

#endif // TECS_IO_DUMMY_HPP
