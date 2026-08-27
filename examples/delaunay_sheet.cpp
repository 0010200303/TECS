// Relax a 2D cell sheet using Delaunay-neighbor forces.
//
// Cells interact only with their Delaunay neighbors — the natural topology
// for a dense monolayer — via adhesion and repulsion.  Unlike all-pairs or
// Gabriel, the Delaunay triangulation captures the correct local neighbourhood
// for a confluent tissue while being far more efficient than all-pairs.
//
// The Delaunay triangulation is recomputed at every time step so that the
// interaction graph evolves correctly as cells move.

#include "../include/tecs.hpp"

using namespace tecs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_PAIR_FINDER(pair_finders::NaiveDelaunay)
  CONFIG_DIMENSIONS(2)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 100;
const int steps = 50;
const double dt = 0.1;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/delaunay_sheet");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * 8;

  Simulation<SimulationConfig> sim(settings);
  auto& links = sim.get_links();
  sim.init_random_disk(1.0);

  DeviceVar<int> link_counter = links.get_active_count();
  auto generate_links = PAIRWISE_FORCE(
    if (is_full_step == false)
      return;

    int link_n = Kokkos::atomic_fetch_add(link_counter.data(), 1);
    links(link_n) = Link(i, j);
  );
  sim.take_step(0.0, generate_links);
  links.set_active_count(link_counter);
  sim.write(0.0);

  auto sheet_forces = PAIRWISE_FORCE(
    ctx.position.delta += forces::PiecewiseLinear(displacement, distance, Scalar(0.5), Scalar(0.7));
  );

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, sheet_forces());
    sim.write(i * dt);
  }

  return 0;
}
