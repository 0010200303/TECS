#ifndef TECS_IO_HDF5_READER_HPP
#define TECS_IO_HDF5_READER_HPP

#include <string>

#include <highfive/highfive.hpp>

#include "../types/view.hpp"
#include "view.hpp"

namespace tecs::io {

  /**
   * @brief Reads simulation data back from HDF5 files written by HDF5_Writer.
   *
   * This class lets you load previously saved simulation snapshots back into
   * View objects, for example to inspect results or restart a simulation.
   *
   * **Basic usage:**
   * @code
   *   // Open a saved HDF5 file
   *   tecs::io::HDF5_Reader reader("output/sim_data.h5");
   *
   *   // Read positions from the first time step
   *   auto positions = reader.read_dataset("t0/positions", positions_view);
   * @endcode
   */
  class HDF5_Reader {
    public:
      /**
       * @brief Open an existing HDF5 file for reading.
       *
       * @param path Path to the `.h5` file.
       */
      HDF5_Reader(const std::string& path) : file(path, HighFive::File::AccessMode::ReadOnly) { }

    private:
      HighFive::File file;

    public:
      /**
       * @brief Get the dimensions of a dataset inside the file.
       *
       * Useful for checking the shape of a dataset before reading it,
       * so you can allocate the right-sized View.
       *
       * @param dataset_name Full path to the dataset
       * @return A vector where each entry is the size of one dimension.
       */
      std::vector<size_t> get_dataset_dimensions(const std::string& dataset_name) {
        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        return dataset.getSpace().getDimensions();
      }

      /**
       * @brief Read a dataset from the file into a View.
       *
       * The View must already have the correct size (use
       * `get_dataset_dimensions()` to check). Data is read into host
       * memory and then synced to the device automatically.
       *
       * @tparam T The element type of the View (can be auto inferred).
       * @param dataset_name  Full path to the dataset to read.
       * @param out_view      The destination View (pre-allocated).
       */
      template<typename T>
      void read_dataset(const std::string& dataset_name, View<T> out_view) {
        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        dataset.read(const_cast<typename View<T>::t_host&>(out_view.view_host()));
        out_view.unconditional_sync_device();
      }
  };
} // namespace tecs::io

#endif // TECS_IO_HDF5_READER_HPP
