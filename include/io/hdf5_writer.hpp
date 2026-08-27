#ifndef TECS_IO_HDF5_WRITER_HPP
#define TECS_IO_HDF5_WRITER_HPP

#include <type_traits>
#include <tuple>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include "../utils/utils.hpp"
#include "../types/view.hpp"
#include "../types/link.hpp"
#include "view.hpp"

namespace tecs::io {

  /**
   * @brief Writes simulation data to HDF5 files with optional XDMF metadata.
   *
   * This class is the main output interface for simulation data in the TECS
   * framework. It takes one or more View objects (containing positions,
   * forces, cell properties, etc.) at each time step and writes them to an
   * HDF5 file. Optionally, it also generates an XDMF file that can be opened
   * directly in visualization tools such as ParaView.
   *
   * @tparam SimulationConfig A configuration struct that defines simulation
   *         parameters (dimensions, scalar, data types, etc.). Use the same config
   *         type as your simulation.
   */
  template<typename SimulationConfig>
  class HDF5_Writer {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

    public:
      /**
       * @brief Configuration options for the HDF5 writer.
       */
      struct Settings {
        /** Whether to generate a companion .xmf file. */
        bool write_xmf = true;

        /**
         * Buffer size in bytes before flushing XDMF output to disk.
         * Larger values reduce file I/O at the cost of memory usage.
         */
        std::size_t buffer_treshold = 65536;
      };

    public:
      /**
       * @brief Opens an HDF5 file and optionally prepares an XDMF XML file.
       *
       * The output path is used as a base name: if you pass `"out/sim"`,
       * the class creates `out/sim.h5` (and `out/sim.xmf` if enabled).
       * Any missing parent directories are created automatically.
       *
       * @param path      File path **without** extension.
       * @param settings  Configuration settings for the writer.
       */
      HDF5_Writer(
        const std::string& path,
        const Settings& settings)
        : write_xmf(settings.write_xmf)
        , buffer_threshold(settings.buffer_treshold) {
        std::filesystem::path _path(path);
        std::filesystem::create_directories(_path.parent_path());

        filename = _path.filename().string();
        h5_file = std::make_unique<HighFive::File>(HighFive::File(path + ".h5", HighFive::File::AccessMode::Truncate));

        if (write_xmf == true)
          init_xmf(path);
      }

      /**
       * @brief Flushes any remaining XDMF data and closes the files.
       */
      ~HDF5_Writer() {
        finalize_xmf();
      }

    private:
      const std::size_t buffer_threshold;

      std::string filename;

      std::unique_ptr<HighFive::File> h5_file;
      std::ofstream xmf_file;
      std::string xmf_buffer;

      std::string xmf_static;

      bool write_xmf;

      // SFINAE helpers to detect compile-time dimensions on types
      template<typename T, typename = void>
      struct has_static_dimensions : std::false_type { };

      template<typename T>
      struct has_static_dimensions<T, std::void_t<decltype(T::dimensions)>> : std::true_type { };

      template<typename T, typename = void>
      struct has_get_dimensions : std::false_type { };

      template<typename T>
      struct has_get_dimensions<T, std::void_t<decltype(std::declval<const T&>().get_dimensions())>> : std::true_type { };

      /**
       * @brief Write a single View into an HDF5 group.
       *
       * Only the active entries (up to `get_active_count()`) are written,
       * so partially filled views produce compact files.
       */
      template<typename T>
      void write_single(HighFive::Group& group, View<T>& view) {
        if (view.get_active_count() == 0)
          return;

        view.sync_host();

        auto sub_host = Kokkos::subview(view.view_host(), Kokkos::make_pair(0u, view.get_active_count()));
        group.createDataSet(view.label(), sub_host);
      }

      /**
       * @brief Start the XDMF XML document and open the temporal grid.
       */
      void init_xmf(const std::string& path) {
        xmf_file.open(path + ".xmf", std::ios::out | std::ios::trunc);
        if (xmf_file.is_open() == false)
          return;

        xmf_buffer += "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\">\n\t<Domain>\n\
\t\t<Grid Name=\"Agents\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
      }

      /**
       * @brief Close the XDMF XML document and flush to disk.
       */
      void finalize_xmf() {
        if (xmf_file.is_open() == false)
          return;
        
        xmf_buffer += "\t\t</Grid>\n\t</Domain>\n</Xdmf>\n";
        xmf_file.write(xmf_buffer.data(), xmf_buffer.size());
        xmf_file.close();
      }

      /**
       * @brief Write the <Grid> header and <Geometry> block for one time step.
       *
       * The first View passed to write() is treated as the position data
       * and becomes the XDMF geometry. The geometry type (X, XY, or XYZ)
       * is chosen based on the simulation's dimension setting.
       */
      template<typename T>
      void write_xmf_grid_start(
        const double time,
        const unsigned int step,
        const View<T>& first_view,
        const bool has_links
      ) {
        xmf_buffer += "\t\t\t<Grid Name=\"t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "\" GridType=\"Uniform\">\n\t\t\t\t<Time Value=\"";
        xmf_buffer += std::to_string(time);
        if (has_links == false) {
          xmf_buffer += "\"/>\n\t\t\t\t<Topology TopologyType=\"Polyvertex\" NumberOfElements=\"";
          xmf_buffer += std::to_string(first_view.get_active_count());
        }
        xmf_buffer += "\"/>\n\t\t\t\t<Geometry GeometryType=\"";
        
        if constexpr (dimensions == 1)
          xmf_buffer += "X";
        else if constexpr (dimensions == 2)
          xmf_buffer += "XY";
        else
          xmf_buffer += "XYZ";
        
        xmf_buffer += "\">\n\t\t\t\t\t<DataItem Format=\"HDF\" Dimensions=\"";
        xmf_buffer += std::to_string(first_view.get_active_count());
        xmf_buffer += " ";
        xmf_buffer += std::to_string(SimulationConfig::dimensions);
        xmf_buffer += "\">\n\t\t\t\t\t\t";
        xmf_buffer += filename;
        xmf_buffer += ".h5:t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "/";
        xmf_buffer += first_view.label();
        xmf_buffer += "\n\t\t\t\t\t</DataItem>\n\t\t\t\t</Geometry>\n";
      }

      /**
       * @brief Append an XDMF <Attribute> (or <Topology> for links) for one View.
       *
       * The attribute type (Scalar / Vector / Tensor / Matrix) is inferred
       * automatically from the View's dimensions. Link views produce a
       * polyline topology instead of an attribute.
       */
      template<typename T>
      void write_xmf_grid_item(std::string& buffer, const std::string& group, const View<T>& view) {
        // TODO: optimize this
        if (view.extent(0) == 0)
          return;

        // extract dimensions
        std::vector<int> dimension_map;
        for (int i = 0; i < view.rank(); ++i)
          dimension_map.push_back(view.extent(i));

        if constexpr (has_static_dimensions<T>::value)
          dimension_map.push_back(T::dimensions);
        else if constexpr (has_get_dimensions<T>::value)
          dimension_map.push_back(T{}.get_dimensions());

        // overwrite first dimension with views active count
        dimension_map[0] = view.get_active_count();

        std::string dimensions = "";
        for (const auto& d : dimension_map) {
          dimensions += std::to_string(d);
          dimensions += " ";
        }

        // get attribute type based on dimensions
        std::string attribute_type = "Matrix";
        if (dimension_map.size() == 1)
          attribute_type = "Scalar";
        else if (dimension_map.size() == 2 && (dimension_map[1] == 2 || dimension_map[1] == 3))
          attribute_type = "Vector";
        else if (dimension_map.size() == 2 && dimension_map[1] == 9)
          attribute_type = "Tensor";

        // line or attribute
        if constexpr (std::is_same_v<T, Link> == true) {
          buffer += "\t\t\t\t<Topology TopologyType=\"Polyline\" NumberOfElements=\"";
          buffer += std::to_string(view.get_active_count());
          buffer += "\">\n";
          buffer += "\t\t\t\t\t<DataItem Format=\"HDF\" DataType=\"UInt32\"";
        }
        else {
          buffer += "\t\t\t\t<Attribute Name=\"";
          buffer += view.label();
          buffer += "\" AttributeType=\"";
          buffer += attribute_type;
          buffer += "\" Center=\"Node\">\n";
          buffer += "\t\t\t\t\t<DataItem Format=\"HDF\"";
        }

        buffer += " Dimensions=\"";
        buffer += dimensions;
        buffer += "\">\n\t\t\t\t\t\t";
        buffer += filename;
        buffer += ".h5:";
        buffer += group;
        buffer += "/";
        buffer += view.label();
        buffer += "\n\t\t\t\t\t</DataItem>\n";
        
        // line or attribute
        if constexpr (std::is_same_v<T, Link> == true)
          buffer += "\t\t\t\t</Topology>\n";
        else
          buffer += "\t\t\t\t</Attribute>\n";
      }

      /**
       * @brief Close the current <Grid> block in XDMF.
       *
       * Also appends the static data chunk and flushes the buffer to disk
       * if it exceeds the configured threshold.
       */
      void write_xmf_grid_end() {
        xmf_buffer += xmf_static;
        xmf_buffer += "\t\t\t</Grid>\n";

        if (xmf_buffer.size() > buffer_threshold) {
          xmf_file.write(xmf_buffer.data(), xmf_buffer.size());
          xmf_buffer.clear();
        }
      }

    public:
      /**
       * @brief Write one time step of simulation data.
       *
       * All provided Views are stored in an HDF5 group named `t<step>`
       * and (if XDMF is enabled) listed as attributes of a new grid entry.
       *
       * @note The **first** View must contain position coordinates – it
       *       becomes the geometry in the XDMF file. Any Link views among
       *       the remaining arguments are rendered as polyline topology
       *       instead of regular attributes.
       *
       * @param time        The simulation time for this step.
       * @param step        The step index (used for naming the output group).
       * @param first_view  The position View (becomes XDMF geometry).
       * @param rest_views  Additional property Views (forces, velocities, links, ...).
       */
      template<typename T0, typename... Ts>
      void write(const double time, const unsigned int step, View<T0>& first_view, View<Ts>&... rest_views) {
        HighFive::Group group = h5_file->createGroup(std::string("t") + std::to_string(step));
        write_single(group, first_view);
        (write_single(group, rest_views), ...);

        h5_file->flush();

        if (write_xmf == false || xmf_file.is_open() == false)
          return;

        bool has_links = ((std::is_same_v<Ts, Link> && rest_views.get_active_count() != 0) || ...);
        write_xmf_grid_start(time, step, first_view, has_links);
        (write_xmf_grid_item(xmf_buffer, "t" + std::to_string(step), rest_views), ...);
        write_xmf_grid_end();
      }

      /**
       * @brief Write data that does not change over time (e.g. a fixed cell type).
       *
       * Static data is stored once in a `"static"` HDF5 group and is
       * included in every XDMF time step so that visualisation tools
       * can display it alongside the time-varying data.
       *
       * @note This method may only be called **once**. Calling it a
       *       second time throws a `std::runtime_error`.
       *
       * @param static_views One or more Views containing static data.
       */
      template<typename... Ts>
      void write_static(View<Ts>&... static_views) {
        if (xmf_static.empty() == false)
          throw std::runtime_error("Static data has already been written. You can only write to static data once.");

        HighFive::Group static_group = h5_file->createGroup("static");
        (write_single(static_group, static_views), ...);

        h5_file->flush();

        if (write_xmf == false || xmf_file.is_open() == false)
          return;

        (write_xmf_grid_item(xmf_static, "static", static_views), ...);
      }
  };
} // namespace tecs::io

#endif // TECS_IO_HDF5_WRITER_HPP
