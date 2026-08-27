#include "../../include/tecs.hpp"

enum CellType {
  Serosa,
  Anchor,
  Pole,
  Embryo
};

using namespace tecs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::BinnedGabriel)
  CONFIG_COM_FIXER(com_fixers::NoComFixer)
};
EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

const double dt = 0.001;
const int steps = 100;
const int steps_per_reduction = 500;
Scalar r_max = 1.0f;
Scalar r_min = 0.0f;

int main(int argc, char* argv[]) {
  // argparse settings
  std::string output_path;
  std::string input_path;
  std::string surface_path;
  bool ok = Arguments("tribolium")
    .add_argument("-rmax", "--rmax", r_max, 1.0f)
    .add_argument("-rmin", "--rmin", r_min, 0.0f)
    .add_argument("-o", "--output", output_path, "./output/serosa")
    .add_argument("-i", "--input", input_path, "./examples/tribolium/initial_data/serosa.h5")
    .add_argument("-s", "--surface", surface_path, "./examples/tribolium/initial_data/surface.h5")
    .parse(argc, argv);
  
  if (ok == false)
    return 1;
  std::cout << "==================== SIMULATION CONFIG ====================" << "\n";
  std::cout << "\tr_max: " << r_max << "\n";
  std::cout << "\tr_min: " << r_min << "\n";
  std::cout << "\toutput path: " << output_path << "\n";
  std::cout << "\tinput path: " << input_path << "\n";
  std::cout << "\tsurface path: " << surface_path << "\n";
  std::cout << "===========================================================" << std::endl;

  /*
  * The cells need a suface acting as an outward force to pretend the shape to collapse
  * Therefore the surface and the normal vectors for the surface points is loaded
  * The surface has about 10 times the number of points then cells in the system
  */

  // prepare readers
  io::HDF5_Reader serosa_reader(input_path);
  io::HDF5_Reader surface_reader(surface_path);

  // create simulation
  PairFinder::Settings pair_finder_settings;
  pair_finder_settings.rebuild_every_n = 20;
  Simulation<SimulationConfig> sim(
    serosa_reader.get_dataset_dimensions("POINTS")[0],
    output_path,
    r_max,
    pair_finder_settings
  );

  // read data
  auto positions_view = sim.get_view<FIELD(Vector, position)>();
  serosa_reader.read_dataset("POINTS", positions_view);

  View<int> cell_types_view("cell_types_view", sim.get_agent_count());
  serosa_reader.read_dataset("cell_type", cell_types_view);

  View<Vector> normals_view("normals_view", surface_reader.get_dataset_dimensions("POINTS")[0]);
  surface_reader.read_dataset("POINTS", normals_view);

  View<Vector> grid_view("grid_view", normals_view.get_capacity());
  for (int i = 0; i < normals_view.get_capacity(); ++i)
    grid_view(i) = 15 * normals_view(i);
  grid_view.auto_sync();

  sim.write(0.0, cell_types_view);

  // simulation loop
  acceleration::Grid<Vector> grid(grid_view);
  grid.rebuild();
  auto system_forces = GENERIC_FORCE(
    int nearest_grid_point_idx = grid.get_nearest_point_index(ctx.position.self);
    if (nearest_grid_point_idx >= 0) {
      Vector relative_position = ctx.position.self - grid_view(nearest_grid_point_idx);
      Vector normal = normals_view(nearest_grid_point_idx);

      Scalar F = -12.0f * relative_position.dot(normal);
      ctx.position.delta += F * normal;
    }
  );

  auto force_between_cells = PAIRWISE_FORCE(
    const int this_type = cell_types_view(i);
    const Scalar exp_neg_d = Kokkos::exp(-distance);

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      ctx.position.delta += (exp_neg_d - 0.5f) * displacement;
    }
    else {
      if (cell_types_view(j) == CellType::Pole)
        ctx.position.delta += (-exp_neg_d) * displacement;
      ctx.position.delta += (exp_neg_d - 0.4f) * displacement;
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      drag += 1.0;
  );

  // pre-allocate vectors for host-side cell removal
  std::vector<Vector> pole_positions;
  pole_positions.reserve(sim.get_agent_count() / 10);

  for (int i = 0; i < steps; ++i) {
    std::cout << sim.get_agent_count() << " step " << i << "\n";

    for (int j = 0; j < steps_per_reduction; ++j)
      sim.take_step(dt, force_between_cells(), system_forces());

    // remove embryo cells near poles
    positions_view.auto_sync();
    cell_types_view.auto_sync();

    pole_positions.clear();
    for (int k = 0; k < sim.get_agent_count(); ++k) {
      if (cell_types_view(k) == CellType::Pole)
        pole_positions.push_back(positions_view(k));
    }

    int new_cell_count = 0;
    for (int k = 0; k < sim.get_agent_count(); ++k) {
      Vector position_k = positions_view(k);
      int cell_type_k = cell_types_view(k);

      bool keep = true;
      if (cell_type_k == CellType::Embryo) {
        for (const auto& pole_position : pole_positions) {
          if (position_k.distance_to_squared(pole_position) < 0.25f) {
            keep = false;
            break;
          }
        }
      }

      if (keep == true) {
        positions_view(new_cell_count) = position_k;
        cell_types_view(new_cell_count) = cell_type_k;
        ++new_cell_count;
      }
    }
    positions_view.auto_sync();
    cell_types_view.auto_sync();
    sim.set_agent_count(new_cell_count);

    sim.write(i * dt, cell_types_view);
  }

  return 0;
}
